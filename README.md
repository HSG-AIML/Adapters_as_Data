# Adapters as Data

This repository contains the code for the workshop paper "Adapters as Data: The Latent Space of LoRAs" for the NeurIPS 2026 Workshop on Neural Network Artifacts as a New Data Modality.


## Introduction

LoRA adapters are usually treated as finished artifacts rather than as a data modality. Prior work has shown that LoRA adapters populate a low-dimensional manifold that a linear PCA basis already organizes well enough to support different operations such as sampling, editing, and inversion. We ask what a learned, nonlinear latent adds beyond this. We train a transformer autoencoder on the same population of LoRA identity adapters, mapping each into a latent space and compare every operation against the identical operation applied directly to the raw adapter weights. The latent preserves the capabilities already available in weight space, sampling diverse identities and editing individual attributes. Beyond preservation, we show that the latent is far more robust to displacement than weight space, degrading into distorted but still valid faces rather than texture. Fixed-dimensional weight-space representations cannot natively host adapters of more than one rank, so we extend our sequential autoencoder to hold mixed-rank adapters within one shared latent. This gives a path toward treating adapter weights of varying architecture as a single, general modality rather than one tied to a fixed rank, and points to a learned, nonlinear latent as the mechanism that could carry this generalization further, toward heterogeneous populations of adapters.


## Repository layout

This repository specializes the SANE codebase to populations of LoRA adapters: a transformer autoencoder is trained to compress flattened LoRA weights into a sequence of latent tokens and to decode them back.

**Everything about the post-training experiments — the artifact bridge, sampling, arithmetic, interpolation, editing, perturbation, reconstruction cycles, rendering, and the tokenization figures — is documented in [experiments/README.md](./experiments/README.md).** This file covers the SANE architecture and how to train it.

`run.py` composes the Hydra config from `config/` and hands it to the
orchestrator. `experiments/` holds the post-training work, `data/` the weight
populations the configs point at (not checked in). Inside `src/sane/`:

| Module | Contents |
|---|---|
| `orchestrator.py` | Resolves the trainer and runner from the config, saves it, launches the run. |
| `trainer/` | The `Trainer` protocol (`base.py`) and `SANETrainer` (`trainer.py`). |
| `runner/` | The `Runner` protocol and `RunResult` (`base.py`), `LocalRunner`, `RayRunner`. |
| `model/` | The autoencoder and its parts: transformers, MLPs, projection heads, translator, position embeddings. |
| `loss/` | `SANELoss` base and `MSEReconstructionLoss`. |
| `data/` | Datasets, tokenizers, windowing, disk cache, splitters, and the dataset factory. |
| `lora/` | Converting flat LoRA vectors back into PEFT state dictionaries. |
| `logging/` | Metrics backends behind a common protocol: JSON, TensorBoard, console. |
| `postprocessing.py` | Shared loading and decoding helpers used by `experiments/`. |
| `utils/` | Activation lookup, weight initialisation, memory logging, tensor inspection. |

## Architecture overview

Three abstractions separate **what** to train from **how** to execute it:

```
run.py  (entry point — Hydra config composition)
  └─► Orchestrator  (resolves trainer/runner, saves config, launches)
        └─► Runner  (training loop, checkpoint schedule, grid search, logging)
              └─► Trainer  (pure PyTorch — model, data, forward/backward, optimizer step)
```

| Component | Owns | Does NOT own |
|---|---|---|
| **Trainer** | Model, data, optimizer, loss, forward/backward pass | Logging, checkpoint schedule, grid search, Ray |
| **Runner** | Training loop, checkpoint schedule, metrics logging, grid search | Config resolution, stage sequencing |
| **Orchestrator** | Resolving trainer/runner from config, wiring, launching | Training mechanics, execution backend details |

---

## Data pipeline

The input is a population of LoRA adapters, each already flattened to a single
vector. `DatasetFactory.build(...)` assembles four stages from one config and
returns train, validation, and test loaders:

```text
data/all_weights.pt              [N, D] flat adapters
  └─► LoRAWeightsDataset         memory-mapped, filtered, cast    -> {"vector": [D]}
        └─► tokenizer            chunk into tokens                -> tokens, mask, position
              └─► CachedWindowedDataset   cut windows, cache to disk
                    └─► DataLoader        (tokens, mask, position) batches
```

`LoRAWeightsDataset` memory-maps the weights file so resident memory stays small,
drops adapters with NaN or above `max_absolute_weight`, caps the population with
`max_adapters`, casts to `dtype`, and caches the valid-index mask so the filter
scan happens once. Setting `class: "CombinedLoRAWeightsDataset"` concatenates
several such datasets — the rank-1 and rank-2 populations trained together —
each tagged with its own `rank_label`. `SequentialSplitter` cuts by the `split`
proportions, per source for a combined dataset so both ranks keep their
proportions in every split.

### Tokenizers

Both tokenizers emit the SANE triple: `tokens [n, tokensize]`, `mask
[n, tokensize]` (True where valid), and `position [n, 3]` holding
`(0, matrix index, token index within that matrix)`.

- **`vector`** treats the adapter as one contiguous stream and chunks it into
  fixed-width tokens, so only the final token is padded. Dense and compact, but a
  token can straddle two weight matrices.
- **`sparse_vector`** respects matrix boundaries: each matrix is split into
  `ceil(numel / tokensize)` tokens and only its last token is padded, so no token
  mixes two matrices. It needs a `weight_dimensions_file` — an ordered
  `{key: [flat_shape, real_shape]}` map of the layout the population was
  flattened with — which also lets it rebuild a state dict on detokenization.
  More tokens carry padding, in exchange for tokens that align with the model's
  structure.

`experiments/tokenization/` renders both schemes as figures.

### Windowing and caching

`WindowedDataset` cuts a token sequence into windows of `window_size` and returns
one per item. `CachedWindowedDataset` adds a disk cache keyed by a hash of the
resolved data config, so tokenizing and windowing happen once per configuration
rather than once per epoch; `cache_mode: "per_window"` stores each window
separately, and the cache is preloaded at construction to keep DataLoader workers
from racing. With `num_windows_per_model: 1` and a `window_size` at least as
large as the token count, each adapter is a single full-coverage window — the
setting all shipped configs use.

---

## Model

`SANEAutoEncoder` (`model/autoencoder.py`) is assembled from five slots by
`SANEAutoEncoder.from_config(config["model"])`:

| Slot | Purpose | Available |
|---|---|---|
| `encoder` | Tokens `[B, n, input_dim]` to latents `[B, n, latent_dim]` | `transformer` |
| `decoder` | Latents back to tokens | `"symmetric"` (mirror of the encoder) or an explicit block |
| `embedder` | Pools latent tokens into one embedding | `average` |
| `projector` | Projection head on the embedding | `mlp`, `average` |
| `translator` | Optional latent-to-latent map | `MLPTranslator`, or `null` |

The forward pass takes `x`, `mask`, and `p` and returns the dict the loss
consumes; `encoder` and `decoder` are also callable on their own, which is what
the post-training experiments use.

`SANETransformerEncoder` wraps either a GPT-style stack (`transformer_type:
"gpt"`, the shipped choice, with GPT-2 style initialisation) or PyTorch's own
encoder layers, with `learned` or sinusoidal position embeddings over the
three-column positions; `max_positions: "auto"` sizes the learned tables from the
data. `SANEMLPAutoEncoder` (`architecture: "mlp_autoencoder"`) is the
non-transformer alternative: residual MLP blocks over a single token holding the
whole adapter.

`MSEReconstructionLoss` computes masked MSE between `x` and `x_hat` and reports
an R² score alongside it. `normalize` selects the normalisation — `null`,
`"token"`, `"window"`, or `"batch"` — and multiple views are averaged.

---

## Trainers

Trainers hold the ML logic. They are plain classes: no base class to inherit,
only the `Trainer` protocol (structural subtyping,
[spec](https://typing.python.org/en/latest/spec/protocol.html)):

```python
class Trainer(Protocol):
    epoch: int
    def setup(self, config: dict[str, Any]) -> None: ...
    def step(self) -> dict[str, Any]: ...
    def save_checkpoint(self, checkpoint_dir: str) -> None: ...
    def load_checkpoint(self, checkpoint_dir: str) -> None: ...
```

`setup` is called once by the runner; `step` runs one epoch and returns a flat
dict of scalar metrics, always including `"loss"` — both runners pick the best
grid-search trial by it. A trainer never imports `ray`, `wandb`, or a logging
backend; it returns metrics and the runner forwards them.

`SANETrainer` is the one trainer in the repo (`trainer: "SANETrainer"`). Its
`setup` seeds torch, builds the dataloaders through `DatasetFactory`, infers
`n_tokens` and `max_positions` from the first batch when they are `"auto"`,
constructs the autoencoder, and resolves the optimizer and scheduler by class
name from `torch.optim` and `torch.optim.lr_scheduler`. Each `step()` runs one
training epoch and, every `validation_frequency` epochs, a validation pass over
the test and validation loaders, suffixing those metrics with `_test` and `_val`.
AMP (`use_amp`, `amp_dtype`), gradient clipping (`max_grad_norm`), and memory
monitoring (`monitor_memory`) are config-driven; when a batch carries multiple
views, `view_1_canon_train` and `view_1_canon_valid` choose between the canonical
view and a random one.

`save_checkpoint` writes one `checkpoint.pt` per checkpoint directory holding
`epoch`, the resolved `config`, and the `model`, `optimizer`, `scheduler`, and
`scaler` state dicts. That file is what `experiments/` loads.

---

## Runners

Runners own the loop and everything about execution — checkpoint schedule,
metrics logging, grid expansion — behind the `Runner` protocol:

```python
def run(
    self,
    trainer_class: type,
    config: dict[str, Any],
    epochs: int,
    checkpoint_frequency: int,
    storage_dir: Path,
) -> RunResult: ...
```

`RunResult` carries the best `checkpoint_path`, the resolved `config`, and the
final `metrics`.

**`LocalRunner`** runs in the current process: it expands grid specs, runs the
trials sequentially, and returns the best by `loss`. Each trial gets
`<storage_dir>/<TrainerClass>_<timestamp>/` holding `trial.yaml`, `config.json`,
and `checkpoint_<epoch:06d>/checkpoint.pt`, written every `checkpoint_frequency`
epochs and at the end. Without an injected logger it builds a `CompositeLogger`
over the JSON, TensorBoard, and console backends.

**`RayRunner`** wraps the trainer as a Ray Tune function trainable. `{"grid":
[...]}` specs become `tune.grid_search(...)`, trials run under `tune.Tuner` with
`num_to_keep` checkpoints retained, and the best result by `loss` is returned.
Setting `wandb_project` adds W&B reporting; setting `resume_path` restores a
previous Tuner instead of starting fresh.

## Orchestrator

The glue between config, runner, and trainer. It creates
`<root_dir>/<experiment_name>/` and saves the resolved config there as
`config_<hash>.yaml` (hashed over everything but `root_dir`), resolves the
trainer class, builds the runner honouring `--debug`, then delegates to
`runner.run(...)` and returns its `RunResult`. Trainers and runners are named by
short alias (`"SANETrainer"`, `"local"`, `"ray"`) or by full dotpath; a
`_target_` key inside a sub-dict works too.

---

# Running the code

## Basic usage

Every run needs a root config (`-cn`) and a `root_dir` to write into:

```bash
python run.py -cn celeb_lora root_dir=/path/to/runs
python run.py -cn celeb_lora_sparse root_dir=/path/to/runs --debug
python run.py -cn celeb_lora root_dir=/path/to/runs runner=local epochs=10
```

Any config value can be overridden on the command line the usual Hydra way.

### Shipped root configs

| Config | Tokenizer | Token size | Window | Model |
|---|---|---|---|---|
| `celeb_lora` | `vector` (dense) | 320 | 312 | transformer, `d_model` 1024, 16 heads, 8 layers |
| `celeb_lora_sparse` | `sparse_vector` | 768 | 170 | transformer, `d_model` 768, 12 heads, 6 layers |
| `celeb_lora_combined_sparse` | `sparse_vector` | 640 | 321 | transformer, `d_model` 640, 10 heads, 6 layers |
| `celeb_lora_mlp` | `vector` (one token) | 99648 | 1 | residual MLP autoencoder |

All four train `SANETrainer` with `MSEReconstructionLoss`, AdamW, and OneCycleLR
into a 256-dimensional latent. `celeb_lora_combined_sparse` trains the rank-1 and
rank-2 populations together on a unified token size.

### Input files

The `checkpoints_dataset` and `tokenizer` configs point at a `data/` directory
beside `run.py`:

| File | Used by |
|---|---|
| `data/all_weights.pt` | `celeb_lora`, `celeb_lora_sparse`, `celeb_lora_mlp` |
| `data/all_weights_r1_padded.pt` | `celeb_lora_combined_sparse`; **not included in the download** — create from `all_weights.pt` by running `python experiments/pad_weights.py` once |
| `data/all_weights_r2.pt` | `celeb_lora_combined_sparse` |
| `data/weight_dimensions.pt` | the `sparse_vector` tokenizer |
| `data/weight_dimensions_unified.pt` | the unified rank-1 + rank-2 tokenizer |

Download the files other than `all_weights_r1_padded.pt`
[here](https://doi.org/10.5281/zenodo.23393961) and place them in `data/` beside
`run.py`. Then run `python experiments/pad_weights.py` from the repository root
to generate the padded rank-1 file. No layout files are needed. The script
preserves the original file and refuses to overwrite an existing output;
allow approximately 96 GiB of free disk space for temporary storage and the
final padded file.

Point `data.checkpoints_dataset.weights_path` and
`data.tokenizer.weight_dimensions_file` elsewhere to use another population.
`config/data/tokens_dataset/*.yaml` also sets `cache_dir` for the window cache;
give it a path with room for the tokenized dataset.

## Choosing a runner

Set `runner: "local"` or `runner: "ray"` in the config, or pass `--debug` to
force local execution regardless:

```bash
python run.py -cn celeb_lora root_dir=/path/to/runs --debug   # forces LocalRunner
python run.py -cn celeb_lora root_dir=/path/to/runs runner=local
```

`_base_.yaml` sets `runner: "ray"`, so a plain run uses Ray. It also holds Ray's
resources — `num_cpus`, `num_gpus`, `cpu_per_trial`, `gpu_per_trial`,
`num_samples`, `num_checkpoints_to_keep`, `ray_verbose`, and the `wandb_*` keys —
so size the trials to your machine there or on the command line:

```bash
python run.py -cn celeb_lora root_dir=/path/to/runs \
  num_gpus=4 gpu_per_trial=1 cpu_per_trial=8
```

## Logging

`log_level` sets the level of the `sane` logger, and Hydra can raise it per
module:

```bash
python run.py -cn celeb_lora root_dir=/path/to/runs \
  log_level=ERROR hydra.job_logging.loggers.sane.model.level=DEBUG
```

With that, you can set the global log level to ERROR and keep the modules of
interest in DEBUG.

## Grid search

Define parameter grids in YAML with the `{grid: [...]}` syntax, at any depth:

```yaml
loss:
  class: "MSEReconstructionLoss"
  normalize: {grid: ['token', 'batch', null]}

optimizer:
  class: "AdamW"
  lr: {grid: [1e-4, 1e-3]}
```

This runs `3 × 2 = 6` trials. With `runner: "local"` they run sequentially; with
`runner: "ray"` they become `tune.grid_search` specs and run in parallel via Ray
Tune. Either way the best trial by `loss` comes back in the `RunResult`.
