import logging

from torch.utils.data import DataLoader, Subset, ConcatDataset

from sane.data import datasets
from sane.data.checkpoint.checkpoint import Checkpoint

from sane.data.datasets import (
    CheckpointsDataset,
    LoRAWeightsDataset,
)
from sane.data.splitter import SequentialSplitter
from sane.data.tokenizers import VectorTokenizer, SparseVectorTokenizer
from sane.utils.generic_utils import log_mem

from sane.data.config_schema import (

    CachedWindowedDatasetConfig,
    CombinedLoRAWeightsDatasetConfig,
    DataLoaderConfig,
    DatasetFactoryConfig,
    SplitConfig,
    TokenizerConfig,

    LoRAWeightsDatasetConfig,
)

logger = logging.getLogger(__name__)


def _ref_checkpoint(
    dataset: CheckpointsDataset | Subset[CheckpointsDataset] | LoRAWeightsDataset | ConcatDataset,
) -> Checkpoint:
    reference_checkpoint = dataset[0]

    if isinstance(reference_checkpoint, Checkpoint):
        return reference_checkpoint

    if isinstance(reference_checkpoint, list):
        return reference_checkpoint[0]

    return reference_checkpoint


class DatasetFactory:
    def build(
        self, config: DatasetFactoryConfig
    ) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
        log_mem("before loading checkpoints dataset")
        checkpoints_dataset = self._load_checkpoints_dataset(config.checkpoints_dataset)

        if isinstance(checkpoints_dataset, ConcatDataset):
            splits = self._split_checkpoints_stratified(config.split, checkpoints_dataset)
        else:
            splits = self._split_checkpoints(config.split, checkpoints_dataset)

        tokens_datasets = [
            self._build_tokens_dataset(
                config.tokens_dataset,
                config.tokenizer,
                ds,
                split_name,
                checkpoints_dataset.name,
            )
            for split_name, ds in splits
            if ds is not None
        ]
        loaders = self._make_dataloaders(config.dataloader, tokens_datasets)

        split_sizes = {name: len(ds) for name, ds in splits if ds is not None}
        tok = config.tokenizer
        ds_cfg = config.tokens_dataset
        dl_cfg = config.dataloader
        logger.info(
            "Built dataloaders\n"
            "  splits:     %s\n"
            "  tokenizer:  class=%s, mode=%s, tokensize=%d, padding=%s \n"
            "  windows:    size=%d, per_model=%s, overlapping=%s, strategy=%s\n"
            "  dataloader: batch=%d, workers=%d, drop_last=%s",
            split_sizes,
            tok.tokenizer_class, tok.mode, tok.tokensize, tok.padding,
            ds_cfg.window_size, ds_cfg.num_windows_per_model, ds_cfg.allow_overlapping_windows, ds_cfg.window_distribution_strategy,
            dl_cfg.batch_size, dl_cfg.num_workers, dl_cfg.drop_last,
        )
        log_mem("after logging dataloader stats")
        return loaders

    def _load_checkpoints_dataset(
        self,
        config: LoRAWeightsDatasetConfig | CombinedLoRAWeightsDatasetConfig,
    ) -> CheckpointsDataset | ConcatDataset:
        if isinstance(config, CombinedLoRAWeightsDatasetConfig):
            return self._load_combined_lora_weights_dataset(config)
        return self._load_lora_weights_dataset(config)

    def _load_lora_weights_dataset(
        self, config: LoRAWeightsDatasetConfig
    ) -> LoRAWeightsDataset:
        return LoRAWeightsDataset(
            weights_path=config.weights_path,
            drop_nan=config.drop_nan,
            max_absolute_weight=config.max_absolute_weight,
            max_adapters=config.max_adapters,
            rank_label=getattr(config, "rank_label", 1),
        )

    def _load_combined_lora_weights_dataset(
        self, config: CombinedLoRAWeightsDatasetConfig
    ) -> ConcatDataset:
        """Load and combine multiple LoRA weight datasets (e.g. rank 1 + rank 2)."""
        datasets_list = [
            self._load_lora_weights_dataset(ds_cfg)
            for ds_cfg in config.datasets
        ]
        combined = ConcatDataset(datasets_list)

        # ConcatDataset doesn't have .name — add one for logging
        combined.name = "combined_lora_r" + "_r".join(
            str(getattr(ds_cfg, "rank_label", "?"))
            for ds_cfg in config.datasets
        )
        logger.info(
            "Combined LoRA dataset: %d total adapters %s per rank",
            len(combined),
            [len(d) for d in datasets_list],
        )
        return combined


    def _split_checkpoints_stratified(self, config: SplitConfig, combined: ConcatDataset):
        """
        Split each sub-dataset (rank) separately, then concatenate the per-rank
        splits so every split (train/val/test) contains both ranks in the same
        proportions.
        """
        if config.train is None:
            raise ValueError("Train split must be specified.")

        active_splits = [s for s in [config.train, config.val, config.test] if s and s > 0]
        active_names  = [n for n, s in [("train", config.train), ("val", config.val), ("test", config.test)] if s and s > 0]

        # Split each sub-dataset (rank) individually
        per_rank_splits = {name: [] for name in active_names}
        for sub_ds in combined.datasets:
            sub_splits = SequentialSplitter().split(sub_ds, active_splits)
            for name, split in zip(active_names, sub_splits):
                per_rank_splits[name].append(split)

        # Concatenate same-named splits across ranks
        split_map = {}
        for name in active_names:
            split_map[name] = ConcatDataset(per_rank_splits[name])

        full_split_map = dict(zip(["train", "val", "test"], [None, None, None])) | split_map
        return tuple((k, v) for k, v in full_split_map.items())

    def _split_checkpoints(
        self,
        config: SplitConfig,
        dataset: CheckpointsDataset | ConcatDataset,
    ) -> tuple[
        tuple[str, Subset],
        tuple[str, Subset | None],
        tuple[str, Subset | None],
    ]:
        if config.train is None:
            raise ValueError("Train split must be specified.")

        active_splits = [
            s
            for s in [config.train, config.val, config.test]
            if s is not None and s > 0
        ]
        active_names = [
            name
            for name, s in [
                ("train", config.train),
                ("val", config.val),
                ("test", config.test),
            ]
            if s is not None and s > 0
        ]
        splits = SequentialSplitter().split(dataset, active_splits)

        split_map = dict(zip(active_names, splits))
        full_split_map = (
            dict(zip(["train", "val", "test"], [None, None, None])) | split_map
        )
        return tuple((k, v) for k, v in full_split_map.items())  # type: ignore

    def _build_tokens_dataset(
        self,
        config: CachedWindowedDatasetConfig,
        tokenizer_config: TokenizerConfig,
        subset: Subset,
        split_name: str,
        dataset_name: str,
    ):
        checkpoints_dataset = subset
        tokenizer = self._build_tokenizer(
            tokenizer_config, _ref_checkpoint(checkpoints_dataset)
        )

        return datasets.CachedWindowedDataset(
            checkpoints_dataset=checkpoints_dataset,
            tokenizer=tokenizer,
            split_name=split_name,
            dataset_name=dataset_name,
            window_size=config.window_size,
            num_windows_per_model=getattr(config, "num_windows_per_model", None),
            num_windows_per_layer=getattr(config, "num_windows_per_layer", None),
            pad_windows=getattr(config, "pad_windows", True),
            allow_overlapping_windows=getattr(config, "allow_overlapping_windows", False),
            window_distribution_strategy=getattr(config, "window_distribution_strategy", "uniform"),
            cache_dir=getattr(config, "cache_dir", None),
            cache_mode=getattr(config, "cache_mode", None),
            global_cache=getattr(config, "global_cache", False),
            data_config_for_hash=getattr(config, "data_config_for_hash", None),
            preload=getattr(config, "preload", True),
        )

    def _build_tokenizer(self, config: TokenizerConfig, reference_checkpoint):

        if config.tokenizer_class == "vector":
            return VectorTokenizer(
                mode=config.mode,
                tokensize=config.tokensize,
                device=config.device,
                reference_statedict=reference_checkpoint,
                padding=config.padding,
                vector_key=getattr(config, "vector_key", "vector"),
            )

        if config.tokenizer_class == "sparse_vector":
            print(f"Building SparseVectorTokenizer with config: {config}")
            return SparseVectorTokenizer(
                mode=config.mode,
                tokensize=config.tokensize,
                device=config.device,
                reference_statedict=reference_checkpoint,
                padding=config.padding,
                vector_key=getattr(config, "vector_key", "vector"),
                weight_dimensions_file=config.weight_dimensions_file,
            )

        raise ValueError(f"Unsupported tokenizer_class: {config.tokenizer_class}")

    def _make_dataloaders(
        self,
        config: DataLoaderConfig,
        tokens_datasets: list[datasets.CachedWindowedDataset],
    ) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
        kwargs: dict = {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
            "drop_last": config.drop_last,
        }
        if config.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = config.prefetch_factor

        shuffles = [False, False, False]
        loaders = {
            ds.split_name: DataLoader(ds, shuffle=s, **kwargs)
            for ds, s in zip(tokens_datasets, shuffles)
        }

        return loaders.get("train"), loaders.get("val", None), loaders.get("test", None)  # type: ignore