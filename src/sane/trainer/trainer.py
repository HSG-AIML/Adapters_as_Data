import copy
import sys
import torch
from torch.amp import autocast, GradScaler
import psutil
import time
import random
import numpy as np
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Optional, Dict, Any, Tuple

from wandb import config



import sane
from sane import loss
from sane.model.autoencoder import SANEAutoEncoder
from sane.data.datasets import *
from sane.data.splitter import *
from sane.loss import *
from sane.utils.torch_utils import batch_info, model_size_info
from sane.utils.generic_utils import log_mem
from sane.data.config_schema import (
    CachedWindowedDatasetConfig,
    DataLoaderConfig,
    DatasetFactoryConfig,
    LoRAWeightsDatasetConfig,
    CombinedLoRAWeightsDatasetConfig,
    SplitConfig,
    TokenizerConfig,
    from_dict,
)
from sane.data.dataset_factory import DatasetFactory
import logging
logger = logging.getLogger(__name__)


class SANETrainer:

    # ----------------------------------- Setup methods ---------------------------------- #

    def __init__(self) -> None:
        pass

    def setup(self, config: Dict[str, Any]):
        """Initializes the trainer with the given configuration."""
        logger.info("Setting up the trainer with configuration: %s", config)

        self.epoch = 0
        log_mem("after initializing trainer")


        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.monitor_memory = config.get("monitor_memory", True)

        # Set the frequency at which validation is performed, in epochs
        # If set to None or 0, validation is not performed
        self.validation_frequency = config.get("validation_frequency", 1)
        if self.validation_frequency == 0:
            self.validation_frequency = None
        elif self.validation_frequency < 0:
            raise ValueError("Validation frequency must be a non-negative integer, but is {}.".format(self.validation_frequency))

        # Load seed and initialize all RNGs
        self.seed = config.get("seed", None)
        if self.seed is not None:
            self._setup_seed(self.seed)


        # How to manage the different views, if need be
        view_1_canon = self.config.get('view_1_canon', False)
        assert isinstance(view_1_canon, bool), "`view_1_canon` must be a boolean value but is {}.".format(type(view_1_canon))

        self.view_1_canon_train = self.config.get('view_1_canon_train', view_1_canon)
        assert isinstance(self.view_1_canon_train, bool), "`view_1_canon_train` must be a boolean value but is {}.".format(type(self.view_1_canon_train))

        self.view_1_canon_valid = self.config.get('view_1_canon_valid', view_1_canon)
        assert isinstance(self.view_1_canon_valid, bool), "`view_1_canon_valid` must be a boolean value but is {}.".format(type(self.view_1_canon_valid))

        # Load the data
        self._load_data()

        # Reset the seed after loading the data to ensure reproducibility
        if self.seed is not None:
            self._setup_seed(self.seed)

        # Initialize the model
        if 'model' not in config: 
            raise ValueError("Configuration must contain a 'model' key with the model configuration.")

        config['model']['device'] = self.device

        # Determine architecture and dispatch accordingly
        architecture = config['model'].get('architecture', 'transformer')

        if architecture == 'transformer':
            if config['model'].get('n_tokens') == 'auto':
                config['model']['n_tokens'] = self._get_n_tokens()
                print(f"Computed number of tokens: {self.config['model']['n_tokens']}")
            if config['model'].get('max_positions', None) == 'auto':
                self.config['model']['max_positions'] = self._get_max_positions()
                print(f"Computed maximum positions: {self.config['model']['max_positions']}")
            self.sane_ae = SANEAutoEncoder.from_config(config['model'])
        elif architecture == 'mlp_autoencoder':
            from sane.model.mlp_autoencoder import SANEMLPAutoEncoder
            self.sane_ae = SANEMLPAutoEncoder.from_config(config['model'])
        else:
            raise ValueError(f"Unknown model architecture: {architecture}")

        print("Set up the SANE AutoEncoder: {}".format(model_size_info(self.sane_ae)))
        log_mem("after model construction")


        # Initialize the optimizer
        if 'optimizer' not in config:
            raise ValueError("Configuration must contain an 'optimizer' key with the optimizer configuration.")
        self._setup_optimizer()

        log_mem("after optimizer setup")

        # Initialize the learning rate scheduler if specified
        if 'scheduler' in config and config['scheduler'] is not None:
            self._setup_scheduler()
        else:
            self.scheduler = None

        # Initialize the loss function
        if 'loss' not in config:
            raise ValueError("Configuration must contain a 'loss' key with the loss function configuration.")
        self._setup_loss()

        # Initialize AMP (Automatic Mixed Precision) if enabled
        self.use_amp = config.get('use_amp', False)
        self.amp_dtype = config.get('amp_dtype', 'float16')
        self.max_grad_norm = config.get('max_grad_norm', None)

        if self.use_amp:
            # Validate amp_dtype
            if self.amp_dtype not in ['float16', 'bfloat16']:
                raise ValueError(f"amp_dtype must be 'float16' or 'bfloat16', got {self.amp_dtype}")
            self.amp_dtype_torch = torch.float16 if self.amp_dtype == 'float16' else torch.bfloat16
            # Use device parameter for GradScaler (new API)
            self.scaler = GradScaler(enabled=('cuda' in self.device))
            logger.info(f"AMP enabled with dtype={self.amp_dtype} on device={self.device}")
        else:
            self.scaler = None
            self.amp_dtype_torch = None

        # Set trainable callbacks
        self.callbacks = config.get('callbacks', None)

    def _setup_seed(self, seed):
        """Sets the random seed for reproducibility."""
        # Python random
        random.seed(seed)
        # NumPy
        np.random.seed(seed)
        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _setup_optimizer(self):
        """Sets up the optimizer based on the configuration."""
        # Setting up the optimizer class
        optimizer_config = self.config['optimizer']
        optimizer_class = optimizer_config.get('class', None)
        optimizer_config = {k: v for k, v in optimizer_config.items() if k != 'class'}
        if optimizer_class is None:
            raise ValueError("Optimizer class must be specified in the configuration.")
        elif isinstance(optimizer_class, str):
            optimizer_class = getattr(torch.optim, optimizer_class)

        if not issubclass(optimizer_class, torch.optim.Optimizer):
            raise TypeError("Optimizer class {} must be a subclass of torch.optim.Optimizer.".format(optimizer_class))

        # Separating the parameters that will be weight-decayed and those that won't
        # This is a common practice to avoid weight decay on biases and batch norm parameters
        decay_params = [p for p in self.sane_ae.parameters() if p.dim() >= 2]
        nodecay_params = [p for p in self.sane_ae.parameters() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params}, # Weight-decayed parameters use the default weight decay or the one specified in the optimizer config
            {"params": nodecay_params, "weight_decay": 0.0}, # Non-weight-decayed parameters have weight decay set to 0
        ]

        # Initializing the optimizer
        self.optimizer = optimizer_class(optim_groups, **optimizer_config)

    def _setup_scheduler(self):
        """Sets up the learning rate scheduler based on the configuration."""
        scheduler_config = self.config['scheduler']
        scheduler_class = scheduler_config.get('class', None)
        scheduler_config = {k: v for k, v in scheduler_config.items() if k != 'class'}
        if scheduler_class is None:
            raise ValueError("Scheduler class must be specified in the configuration.")
        elif isinstance(scheduler_class, str):
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_class)

        if not issubclass(scheduler_class, torch.optim.lr_scheduler.LRScheduler):
            raise TypeError("Scheduler class {} must be a subclass of torch.optim.lr_scheduler._LRScheduler.".format(scheduler_class))

        if scheduler_class == torch.optim.lr_scheduler.OneCycleLR:
            # For OneCycleLR, we need to specify the total number of steps
            scheduler_config['total_steps'] = self._get_total_steps()
            # We also set the max learning rate as the learning rate provided for the optimizer
            scheduler_config['max_lr'] = self.config['optimizer']['lr']

        # Initializing the scheduler
        self.scheduler = scheduler_class(self.optimizer, **scheduler_config)

    def _setup_loss(self):
        """Sets up the loss function based on the configuration."""
        loss_config = self.config.get('loss', {})
        loss_class = loss_config.get('class', None)
        loss_config = {k: v for k, v in loss_config.items() if k != 'class'}
        if loss_class is None:
            raise ValueError("Loss class must be specified in the configuration.")
        elif isinstance(loss_class, str):
            loss_class = getattr(sane.loss, loss_class)

        if not issubclass(loss_class, SANELoss):
            raise TypeError("Loss class {} must be a subclass of SANELoss.".format(loss_class))

        self.loss_fn = loss_class(**loss_config)

    # ----------------------------------- Loading data ----------------------------------- #

    def _load_data(self):
        # This loading can be improved using OmegaConf directly.
        # For now, this is loaded until all configs are cleaned up.
        data_config = self.config["data"]
        chkpt_cfg = data_config["checkpoints_dataset"]
        tok_cfg = data_config["tokenizer"]
        tds_cfg = data_config["tokens_dataset"]
        tds_cfg["data_config_for_hash"] = copy.deepcopy(self.config["data"])
        factory_config = DatasetFactoryConfig(
            checkpoints_dataset=self._resolve_chkpt_cfg(chkpt_cfg),
            tokenizer=from_dict(TokenizerConfig, tok_cfg),
            tokens_dataset=from_dict(CachedWindowedDatasetConfig, tds_cfg),
            split=from_dict(SplitConfig, data_config["split"]),
            dataloader=from_dict(DataLoaderConfig, data_config),
        )
        log_mem("before DatasetFactory.build")

        self.trainloader, self.valloader, self.testloader = DatasetFactory().build(factory_config)
        self.trainset = self.trainloader.dataset
        self.valset = self.valloader.dataset if self.valloader is not None else None
        self.testset = self.testloader.dataset if self.testloader is not None else None

        log_mem("after DatasetFactory.build")




    def _resolve_chkpt_cfg(self, cfg: Any):
        if isinstance(cfg, list):
            return [self._resolve_chkpt_cfg(item) for item in cfg]
        if isinstance(cfg, str):
            from omegaconf import OmegaConf
            config_path = Path(sane.__file__).parent.parent.parent / "config" / "data" / "checkpoints_dataset" / f"{cfg}.yaml"
            if not config_path.exists():
                raise ValueError(f"Checkpoints dataset config '{cfg}' not found at: {config_path}")
            cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

        cls_name = cfg.get("class")
        if cls_name == "LoRAWeightsDataset":
            return from_dict(LoRAWeightsDatasetConfig, cfg)
        if cls_name == "CombinedLoRAWeightsDataset":
            sub_datasets = [
                self._resolve_chkpt_cfg(ds_cfg)
                for ds_cfg in cfg.get("datasets", [])
            ]
            return CombinedLoRAWeightsDatasetConfig(datasets=sub_datasets)
        raise ValueError(f"Unsupported checkpoints dataset class '{cls_name}' in config: {cfg}")


    # --------------------------------- Inferring values --------------------------------- #

    def _get_n_tokens(self) -> int:
        """Returns the number of input tokens, usually equivalent to the window size.
        This is computed as the number of tokens of the first element in the train dataloader.
        If there are multiple number of tokens across the input data, using this automatic n_tokens
        check can lead to errors down the line.

        This is useful for setting the input dimension of the model.

        Returns:
            int: The number of input tokens.
        """
        for batch_idx, data in enumerate(self.trainloader):
            if isinstance(data[0], torch.Tensor):
                return data[0].shape[-2]
            else:
                if not isinstance(data[0][0], torch.Tensor):
                    raise ValueError("Data must be a list of tuples (tokens, mask, position) but is:\n{}.".format(batch_info(data)))
                return data[0][0].shape[-2]

    def _get_max_positions(self) -> List[int]:
        """Returns the maximum position (length) for each dimension of the input position tensors.

        This is useful for the learned position embeddings.

        Returns:
            List[int]: A list containing the maximum position for each input sequence in the batch.
        """
        max_pos = None

        for batch_idx, data in enumerate(self.trainloader):
            if isinstance(data[0], torch.Tensor):
                assert isinstance(data[2], torch.Tensor) and data[2].dtype in [torch.int32, torch.int64]
                batch_max_pos = torch.amax(data[2], dim=tuple(range(data[2].ndim - 1))).int().tolist()
                if max_pos is None:
                    max_pos = batch_max_pos
                else:
                    for i, value in enumerate(max_pos):
                        if batch_max_pos[i] > value:
                            max_pos[i] = batch_max_pos[i]
            else:
                if not isinstance(data[0][0], torch.Tensor):
                    raise ValueError("Data must be a list of tuples (tokens, mask, position) but is:\n{}.".format(batch_info(data)))
                for item in data:
                    assert isinstance(item[2], torch.Tensor) and item[2].dtype in [torch.int32, torch.int64]
                    batch_max_pos = torch.amax(item[2], dim=tuple(range(item[2].ndim - 1))).int().tolist()
                    if max_pos is None:
                        max_pos = batch_max_pos
                    else:
                        for i, value in enumerate(max_pos):
                            if batch_max_pos[i] > value:
                                max_pos[i] = batch_max_pos[i]

            del data

        return [pos+1 for pos in max_pos]

    def _get_total_steps(self) -> int:
        """Returns the total number of training steps for the OneCycleLR scheduler."""
        if not isinstance(self.trainset, CachedWindowedDataset):
            raise ValueError("Trainset must be a CachedWindowedDataset, but is {}.".format(type(self.trainset).__name__))
        return self.config['epochs'] * len(self.trainloader)

    # ----------------------------------- Step methods ----------------------------------- #

    def step(self) -> Dict[str, Any]:
        """Performs a complete iteration step, including training, validation (if scheduled), and memory monitoring.

        This method orchestrates the following actions:
            - Increments the epoch counter.
            - Performs a training step.
            - Performs validation on test and validation datasets if the epoch matches the validation frequency.
            - Monitors memory usage if enabled.

        Returns:
            Dict[str, Any]: A dictionary containing the results of training, validation, and memory monitoring.
        """
        # Increment the epoch counter
        self.epoch += 1

        # Perform a training step
        out_dict = self._step_training()

        # Perform validation if required
        if self.validation_frequency is not None and self.epoch % self.validation_frequency == 0:
            if self.testloader is not None:
                test_perf = self._step_validation(self.testloader)
                out_dict.update({k + '_test': v for k, v in test_perf.items()})
            if self.valloader is not None:
                val_perf = self._step_validation(self.valloader)
                out_dict.update({k + '_val': v for k, v in val_perf.items()})

        # Monitor memory usage if enabled
        if self.monitor_memory:
            out_dict.update(self._step_monitor_memory())

        for key, value in out_dict.items():
            if isinstance(value, torch.Tensor):
                out_dict[key] = value.item()

        return out_dict

    def _step_training(self) -> Dict[str, Any]:
        """Performs a single training step over the entire training dataset.

        Iterates over the training dataset, processes each batch, and computes the loss.
        If the data is a list, it processes each item in the list and averages the results.

        Returns:
            Dict[str, Any]: A dictionary containing the averaged training results and metadata.
        """
        start_time = time.time()
        log_mem(f"train epoch {self.epoch} start")
        # Initialize running sums for efficient accumulation
        running_sum = None
        total_samples = 0

        self.sane_ae.train()

        for batch_idx, data in enumerate(self.trainloader):
            if isinstance(data, Sequence) and isinstance(data[0], Sequence):
                for item in data:
                    batch_result = self._step_training_forward(item, batch_idx=batch_idx)
                    batch_size = item[0].shape[0]

                    # Initialize running_sum on first batch
                    if running_sum is None:
                        running_sum = {k: 0.0 for k in batch_result.keys()}

                    # Accumulate weighted results
                    for k, v in batch_result.items():
                        running_sum[k] += v.item() * batch_size
                    total_samples += batch_size
            else:
                if not isinstance(data, Sequence):
                    raise ValueError("Data must be a sequence of (tokens, mask, position) but is {}.".format(type(data)))
                if len(data) != 3:
                    raise ValueError("Data must be a sequence of (tokens, mask, position) but has length {}.".format(len(data)))

                batch_result = self._step_training_forward(data, batch_idx=batch_idx)
                batch_size = data[0].shape[0]

                # Initialize running_sum on first batch
                if running_sum is None:
                    running_sum = {k: 0.0 for k in batch_result.keys()}

                # Accumulate weighted results
                for k, v in batch_result.items():
                    running_sum[k] += v.item() * batch_size
                total_samples += batch_size

        # Compute final averages
        results = {k: v / total_samples for k, v in running_sum.items()}
        results["epoch"] = self.epoch
        results["training_time"] = time.time() - start_time
        log_mem(f"train epoch {self.epoch} end")
        return results

    def _step_training_forward(self, tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int = -1) -> Dict[str, torch.Tensor]:
        """Performs a forward pass and backpropagation for a single batch of training data.

        Args:
            tensors (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing ``(tokens, mask, position)``.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the computed loss and any additional metrics.
        """
        if len(tensors) != 3:
            raise ValueError("Tensors must be a tuple of (tokens, mask, position) but has length {} and dimensions {}.".format(len(tensors), [tensor.dim() for tensor in tensors]))

        # If the tokens have 4 dimensions, it means we have different views.
        # In that case, dimensions are: (batch_size, num_views, num_tokens, token_size)
        # Whether to take the canonical view or a random view is controlled by the `view_1_canon_train` hyperparameter.
        if len(tensors[0].shape) == 4:
            view_idx = 0 if self.view_1_canon_train else random.randint(0, tensors[0].shape[1] - 1)
            tokens, mask, position = tensors[0][:, view_idx].to(self.device), tensors[1][:, view_idx].to(self.device), tensors[2][:, view_idx].to(self.device)
        # If not using multiple views, just move the tensors to the device
        elif len(tensors[0].shape) == 3:
            tokens, mask, position = [tensor.to(self.device) for tensor in tensors]
        else:
            raise ValueError("Tokens must have 3 or 4 dimensions, but has {}. Shape: {}.".format(tensors[0].dim(), tuple(tensors[0].shape)))

        self.optimizer.zero_grad()

        # Forward pass with optional AMP autocast
        if self.use_amp:
            device_type = 'cuda' if 'cuda' in self.device else 'cpu'
            with autocast(device_type=device_type, dtype=self.amp_dtype_torch):
                output = self.sane_ae(x=tokens, mask=mask, p=position)
                loss = self.loss_fn(output)
        else:
            output = self.sane_ae(x=tokens, mask=mask, p=position)
            loss = self.loss_fn(output)

        # Backward pass with optional gradient scaling
        if self.use_amp:
            # Track scale to detect if optimizer step was skipped due to inf/NaN gradients
            scale_before = self.scaler.get_scale()
            self.scaler.scale(loss['loss']).backward()
            # Optional gradient clipping
            if self.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(self.sane_ae.parameters(), self.max_grad_norm)
                loss['grad_norm'] = total_norm.detach()
                loss['amp_scale'] = torch.tensor(self.scaler.get_scale())
                if total_norm.item() > 30:
                    fp = (tokens.shape, round(tokens.sum().item(), 2),
                        round(tokens.std().item(), 4))
                    print(f"[SPIKE] epoch={self.epoch} grad_norm={total_norm.item():.1f} "
                        f"mask_density={mask.float().mean().item():.3f} "
                        f"token_min={tokens.min().item():.3f} token_max={tokens.max().item():.3f} "
                        f"has_nan={torch.isnan(tokens).any().item()} fingerprint={fp}")
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # Only step scheduler if optimizer actually stepped (scale didn't decrease)
            # see https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step-in-pytorch-1-1-0-and-later-you-should-call-them-in-the-opposite-order-optimizer-step-before-lr-scheduler-step/88295/6
            if self.scheduler is not None:
                scale_after = self.scaler.get_scale()
                if scale_before <= scale_after:
                    self.scheduler.step()
        else:
            loss['loss'].backward()
            # Optional gradient clipping
            if self.max_grad_norm is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(self.sane_ae.parameters(), self.max_grad_norm)
                loss['grad_norm'] = total_norm.detach()
                if total_norm.item() > 30:
                    fp = (tokens.shape, round(tokens.sum().item(), 2),
                        round(tokens.std().item(), 4))
                    print(f"[SPIKE] epoch={self.epoch} grad_norm={total_norm.item():.1f} "
                        f"mask_density={mask.float().mean().item():.3f} "
                        f"token_min={tokens.min().item():.3f} token_max={tokens.max().item():.3f} "
                        f"has_nan={torch.isnan(tokens).any().item()} fingerprint={fp}")
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        return loss

    def _step_validation(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        """Performs a validation step over the provided dataloader.

        Iterates over the validation dataset, processes each batch, and computes the loss.
        If the data is a list, it processes each item in the list and averages the results.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader providing the validation data.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the averaged validation results.
        """
        self.sane_ae.eval()

        # Initialize running sums for efficient accumulation
        running_sum = None
        total_samples = 0

        with torch.no_grad():  # Deactivate gradients computation at validation time
            for data in dataloader:
                if isinstance(data, Sequence) and isinstance(data[0], Sequence):
                    for item in data:
                        batch_result = self._step_validation_forward(item)
                        batch_size = item[0].shape[0]

                        # Initialize running_sum on first batch
                        if running_sum is None:
                            running_sum = {k: 0.0 for k in batch_result.keys()}

                        # Accumulate weighted results
                        for k, v in batch_result.items():
                            running_sum[k] += v.item() * batch_size
                        total_samples += batch_size
                else:
                    if not isinstance(data, Sequence):
                        raise ValueError("Data must be a sequence of (tokens, mask, position) but is {}.".format(type(data)))
                    if len(data) != 3:
                        raise ValueError("Data must be a sequence of (tokens, mask, position) but has length {}.".format(len(data)))

                    batch_result = self._step_validation_forward(data)
                    batch_size = data[0].shape[0]

                    # Initialize running_sum on first batch
                    if running_sum is None:
                        running_sum = {k: 0.0 for k in batch_result.keys()}

                    # Accumulate weighted results
                    for k, v in batch_result.items():
                        running_sum[k] += v.item() * batch_size
                    total_samples += batch_size

        # Compute final averages
        results = {k: v / total_samples for k, v in running_sum.items()}

        return results

    def _step_validation_forward(self, tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Performs a forward pass for a single batch of validation data.

        Args:
            tensors (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing ``(tokens, mask, position)``.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the computed loss and any additional metrics.
        """
        tokens, mask, position = [tensor.to(self.device) for tensor in tensors]

        # If the tokens (or mask, positions) have 4 dimensions, it means we have different views.
        # In that case, dimensions are: (batch_size, num_views, num_tokens, token_size)
        # Whether to take the canonical view or a random view is controlled by the `view_1_canon_valid` hyperparameter.
        if len(tensors[0].shape) == 4:
            view_idx = 0 if self.view_1_canon_valid else random.randint(0, tensors[0].shape[1] - 1)
            tokens, mask, position = tensors[0][:, view_idx].to(self.device), tensors[1][:, view_idx].to(self.device), tensors[2][:, view_idx].to(self.device)
        # If not using multiple views, just move the tensors to the device
        elif len(tensors[0].shape) == 3:
            tokens, mask, position = [tensor.to(self.device) for tensor in tensors]
        else:
            raise ValueError("Tokens must have 3 or 4 dimensions, but has {} (shape: {}).".format(tensors[0].dim(), tuple(tensors[0].shape)))

        # Forward pass with optional AMP autocast (no gradient scaling needed for validation)
        if self.use_amp:
            device_type = 'cuda' if 'cuda' in self.device else 'cpu'
            with autocast(device_type=device_type, dtype=self.amp_dtype_torch):
                output = self.sane_ae(x=tokens, mask=mask, p=position)
                loss = self.loss_fn(output)
        else:
            output = self.sane_ae(x=tokens, mask=mask, p=position)
            loss = self.loss_fn(output)
        return loss

    def _step_monitor_memory(self):
        """Monitors the memory usage during training."""
        # Get CPU memory usage
        process = psutil.Process()
        cpu_mem_info = process.memory_info()
        cpu_mem_usage = {
            'rss': cpu_mem_info.rss,  # Resident Set Size
            'vms': cpu_mem_info.vms,  # Virtual Memory Size
            'percent': psutil.virtual_memory().percent  # Percentage of memory usage
        }

        # Get CUDA memory usage if CUDA is available
        cuda_mem_usage = {}
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            cuda_mem_usage = {
                'free': free,  # Free memory in bytes
                'total': total,  # Total memory in bytes
                'used': total - free,  # Used memory in bytes
                'percent': (total - free) / total * 100  # Percentage of memory usage
            }

        # Combine all memory usage information
        memory_usage = {
            'cpu_memory_usage': cpu_mem_usage,
            'cuda_memory_usage': cuda_mem_usage,
        }

        return memory_usage

    # -------------------------------- Save & Load methods ------------------------------- #

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "checkpoint.pt"

        log_mem(f"Saving checkpoint at epoch {self.epoch} to {checkpoint_path}")

        state = {
            'epoch': self.epoch,
            'config': self.config,
            'model': self.sane_ae.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'scaler': self.scaler.state_dict() if self.scaler else None,
        }

        torch.save(state, checkpoint_path)

        log_mem(f"Checkpoint saved at {checkpoint_path}")


    def load_checkpoint(self, checkpoint_dir: str) -> None:
        checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")

        try:
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")

        self.epoch = state['epoch']
        self.sane_ae.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler and state['scheduler'] is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        if self.scaler and state.get('scaler') is not None:
            self.scaler.load_state_dict(state['scaler'])

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str):
        checkpoint = Path(checkpoint_path) / "checkpoint.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint file {checkpoint} does not exist.")
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint}: {e}")

        config = state.get("config") or _load_trial_config(checkpoint.parent.parent)
        trainer = cls()
        trainer.setup(config)
        trainer.sane_ae.load_state_dict(state['model'])
        trainer.optimizer.load_state_dict(state['optimizer'])
        if trainer.scheduler and state['scheduler'] is not None:
            trainer.scheduler.load_state_dict(state['scheduler'])
        if trainer.scaler and state.get('scaler') is not None:
            trainer.scaler.load_state_dict(state['scaler'])
        return trainer


def _load_trial_config(trial_dir: Path) -> dict:
    """Probe a trial directory for a config file, trying known formats in order.

    Checks: embedded config (handled by caller), ``trial.yaml`` (OmegaConf),
    then ``params.json`` (Ray Tune legacy).
    """
    trial_yaml = trial_dir / "trial.yaml"
    if trial_yaml.exists():
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(OmegaConf.load(str(trial_yaml)), resolve=True)
        return cfg  # type: ignore[return-value]

    params_json = trial_dir / "params.json"
    if params_json.exists():
        config = json.loads(params_json.read_text())
        # Ray Tune wraps the actual config under 'train_loop_config'
        if "train_loop_config" in config:
            config = config["train_loop_config"]
        return config

    raise FileNotFoundError(
        f"No config found in checkpoint or trial directory {trial_dir}. "
        "Expected 'config' key in checkpoint.pt, trial.yaml, or params.json."
    )
