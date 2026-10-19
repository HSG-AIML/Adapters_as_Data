# Experiments

Post-training experiments on a trained SANE checkpoint. Nothing here is called
by `run.py`; every script is a standalone CLI that reads artifacts, writes
artifacts, and records what it did alongside them.

Run any script with `--help` for its full option list.

## Contents

| Path | Purpose |
|---|---|
| `extract_latents.py` | Encode configured dataset splits into latent payloads |
| `reconstruct_latents.py` | Decode latent payloads back into flat LoRA weights |
| `generate_images.py` | Render flat LoRA weights with the frozen diffusion model |
| `sampling/` | Draw new adapters from a fitted Gaussian or PCA model |
| `arithmetic/` | Pairwise arithmetic between two adapters |
| `interpolation/` | Linear interpolation between two adapters |
| `editing/` | Attribute-direction screening and editing |
| `perturbation/` | Gaussian perturbation of adapters |
| `cycle/` | Repeated encode/decode cycles and their evaluation |
| `tokenization/` | Figures of the dense and sparse tokenization schemes |

## The artifact bridge

Every experiment sits between the same two ends: SANE latents `z` on one side,
flat LoRA weights on the other. Three scripts own the crossings.

```text
checkpoint + configured LoRA population
  └─► extract_latents.py ──► latents_<split>.pt          (z, x, p, mask, ids, config)
        └─► <experiment> ──► manipulated latents
              └─► reconstruct_latents.py ──► x_and_x_hat_<name>.pt   (flat x, x_hat)
                    └─► generate_images.py ──► PNG images
```

Weight-space controls skip the middle: they manipulate flat weights directly and
go straight to `generate_images.py`.

```text
reconstruction payload (x)
  └─► <experiment>_weights.py ──► flat x / x_hat ──► generate_images.py
```

### Payload formats

Two payload shapes travel through the pipeline. Both are `torch.save` dicts
tagged with `format` and `format_version`.

`sane.latents` — consumed by `reconstruct_latents.py`:

| Field | Shape | Meaning |
|---|---|---|
| `z` | `[N, tokens, latent_dim]` | Latents, the thing an experiment manipulates |
| `x` | `[N, tokens, features]` | Source token grid, reconstruction metadata |
| `p` | `[N, tokens, 3]` | Token positions |
| `mask` | `[N, tokens]` or `None` | Padding mask |
| `ids` | `[N]` | Stable sample identifiers |
| `config`, `epoch`, `checkpoint` | | Provenance from the source payload |

`sane.reconstructed_weights` — consumed by `generate_images.py`:

| Field | Shape | Meaning |
|---|---|---|
| `x` | `[N, W]` | Original or reference adapter |
| `x_hat` | `[N, W]` | Reconstructed or manipulated adapter |
| `indices` | `[N]` | Stable identifiers; become the image filenames |
| `config`, `epoch`, `checkpoint` | | Provenance |

### Reference `x` is not ground truth

Generated, edited, and sampled payloads still carry an `x` field, because
reconstruction needs a reference token grid for positional ordering and the
renderer expects the field to exist. For those payloads `x` is a *reference*,
not the target of `x_hat`. Every such payload says so in its `reference_note`,
and those outputs must be rendered with `--fields x_hat --no-scores`. Paired
scores are meaningful only for a true reconstruction, where `x` is the adapter
`x_hat` was decoded from.

### Trusted inputs

Every `.pt` input is pickle-backed and is loaded with `weights_only=False`.
Only pass checkpoints, latent payloads, weight populations, and identity
dataframes you produced or trust.

## Setup

```bash
uv sync                                    # experiments that only manipulate tensors
uv sync --extra render                     # plus image rendering
uv sync --extra render --extra evaluation  # plus LPIPS/SSIM scores
```

The base diffusion model is downloaded from Hugging Face on first render. The
renderer also needs the rank-specific PEFT adapter config and tensor layout from
the `data/` directory beside `run.py` — `weight_dimensions.pt` and
`adapter_config.json` for rank 1, `weight_dimensions_unified.pt` and
`adapter_config_r2.json` for rank 2 — or explicit paths via
`--weight-dimensions` and `--adapter-config`. No separate weights2weights
checkout is required.

## Conventions

All scripts share the same shape, so what you learn on one transfers:

- **Positional inputs, named outputs.** The artifact being read is positional;
  `--output` or `--output-dir` is required and explicit.
- **Manifests.** Every script that writes a payload writes a JSON manifest
  beside it recording the resolved inputs, parameters, shapes, and produced
  files. The screen and the cycle evaluator write CSV and JSON results instead;
  the figure scripts write only the figure.
- **`--overwrite`.** A script that writes payloads checks every planned path
  before doing any work and refuses to replace an existing one without this
  flag. The figure scripts and the cycle evaluator overwrite in place.
- **`--device auto`.** Resolves to `cuda:0` when available, else CPU. Present
  on the scripts that run the model or fit in the kernel; the rest are plain
  CPU tensor operations.
- **`--rows`.** Selects row indices into the population. Pairwise experiments
  use `--sample-a` / `--sample-b` instead.
- **`--field x`.** Weight-space scripts accept either a raw `[N, W]` tensor or
  any payload dict; `--field` names the tensor to read from a dict.
- **Matched controls.** Each latent-space experiment has a weight-space twin
  that applies the identical operation to raw weights. Give both the same rows
  and the same parameters, and use `x` from the reconstruction of the same
  latent population so the two runs describe the same adapters.

Suggested output layout, used by every example below:

```text
outputs/
  latents/          extract_latents.py
  reconstructions/  reconstruct_latents.py
  images/           generate_images.py
  <experiment>/     one directory per experiment
```

---

## Bridge scripts

### `extract_latents.py`

Encodes the configured adapter population with a trained checkpoint.

```bash
uv run python experiments/extract_latents.py \
  path/to/checkpoint.pt \
  --output-dir outputs/latents \
  --splits train val test
```

Writes `latents_train.pt`, `latents_val.pt`, `latents_test.pt` and
`extraction_manifest.json`. The checkpoint's embedded config is used by default;
pass `--config` when local dataset paths need overriding.

### `reconstruct_latents.py`

Decodes one or more latent payloads and de-tokenizes them into flat weights.

```bash
uv run python experiments/reconstruct_latents.py \
  path/to/checkpoint.pt \
  outputs/latents/latents_val.pt \
  --output-dir outputs/reconstructions
```

`latents_val.pt` becomes `x_and_x_hat_val.pt`; experiment-specific filenames are
preserved. Accepts several inputs at once, so a directory of manipulated latents
can be decoded in one call. Config precedence is `--config`, then the payload's
embedded config, then the checkpoint's.

### `generate_images.py`

Renders flat adapters through the frozen diffusion model. Requires `--rank`.

```bash
# Paired rendering with weight and image scores
uv run --extra render --extra evaluation python experiments/generate_images.py \
  outputs/reconstructions/x_and_x_hat_val.pt \
  --output-dir outputs/images/reconstructions \
  --rank 1

# Generated or manipulated adapters: x is only a reference
uv run --extra render python experiments/generate_images.py \
  outputs/sampling/weights_pca.pt \
  --output-dir outputs/images/sampling \
  --rank 1 --fields x_hat --no-scores
```

Prompts, seed, inference steps, guidance scale, and resolution are CLI defaults.
Each run writes `render_config.json` beside the PNGs.

---

## Sampling

Draws new adapters from a distribution fitted to a population. Two methods:
`naive` fits an independent Gaussian to every coordinate; `pca` fits a centered
economy-SVD PCA and samples empirical component distributions. `--factor`
multiplies the fitted standard deviations.

```bash
uv run python experiments/sampling/sample_latents.py \
  outputs/latents/latents_train.pt \
  --output outputs/sampling/latents_pca.pt \
  --method pca --components 256 --factor 1 --num-samples 9

uv run python experiments/reconstruct_latents.py \
  path/to/checkpoint.pt outputs/sampling/latents_pca.pt \
  --output-dir outputs/sampling/reconstructed
```

Matched weight-space control, using the same rows via the reconstructed training
payload:

```bash
uv run python experiments/sampling/sample_weights.py \
  outputs/reconstructions/x_and_x_hat_train.pt --field x \
  --output outputs/sampling/weights_pca.pt \
  --method pca --components 256 --factor 1 --num-samples 9 --seed 42
```

Use the same `--method`, `--components`, `--factor`, `--num-samples`, and
`--seed` for both. `--model-output` optionally retains the fitted PCA basis and
projections as a separate artifact.

## Arithmetic

Pairwise operations between two adapters: `diff_a_minus_b`, `diff_b_minus_a`,
`sum`, `mean`, and `add` (`A + scale * B`, expanded over `--add-scales`). One
output file per operation.

```bash
uv run python experiments/arithmetic/arithmetic_latents.py \
  outputs/latents/latents_val.pt \
  --output-dir outputs/arithmetic/latents \
  --sample-a 6 --sample-b 8

uv run python experiments/arithmetic/arithmetic_weights.py \
  outputs/reconstructions/x_and_x_hat_val.pt --field x \
  --output-dir outputs/arithmetic/weights \
  --sample-a 6 --sample-b 8
```

## Interpolation

Evenly spaced linear interpolation between two adapters, endpoints included:
`alpha = 0` is sample A, `alpha = 1` is sample B. Row order within the output is
sweep order.

```bash
uv run python experiments/interpolation/interpolate_latents.py \
  outputs/latents/latents_val.pt \
  --output outputs/interpolation/latents_A2_B13.pt \
  --sample-a 2 --sample-b 13 --steps 10

uv run python experiments/interpolation/interpolate_weights.py \
  outputs/reconstructions/x_and_x_hat_val.pt --field x \
  --output outputs/interpolation/weights_A2_B13.pt \
  --sample-a 2 --sample-b 13 --steps 10
```

## Editing

Fits an attribute direction by ridge regression against binary labels from an
identity dataframe whose row order matches the population, optionally removes a
nuisance attribute's component from it, and applies it as a strength sweep.

Everything runs in the `N x N` kernel: the Gram matrix is built once, all
attributes are solved as one multi-RHS system, and debiasing is done in the dual,
so no `D`-dimensional vector is formed until the directions are materialized.

### Screen first

`screen_directions.py` fits the same direction for every binary attribute and
reports how well the label groups separate, without decoding or rendering
anything. Held-out AUC is the number that predicts whether an edit will work;
the in-sample separation ratio of an interpolating fit is a function of label
prevalence alone.

```bash
uv run python experiments/editing/screen_directions.py \
  outputs/latents/latents_train.pt \
  --identity-df path/to/identity_df.pt \
  --output-dir outputs/editing/screen
```

Writes per-attribute metrics as CSV and JSON, three pgfplots figures and a
booktabs table, and `direction_cosines.csv` — the pairwise cosines that say
which attributes are entangled, and therefore which `--debias` pairs are worth
passing.

### Then edit

```bash
uv run python experiments/editing/edit_latents.py \
  outputs/latents/latents_train.pt \
  --identity-df path/to/identity_df.pt \
  --output-dir outputs/editing/latents \
  --attributes Bald --control-attributes Oval_Face \
  --debias Bald=No_Beard,Male \
  --rows 1
```

`--attributes` and `--control-attributes` are fitted identically; the split only
records which attributes the screen predicted would work. Rendering both groups
is what validates the screen: if the control group shows no visible change at the
strength where the target group clearly works, the screen predicts edit quality.

Latent strengths are relative by default — `alpha` is a fraction of `||z||`.
Weight strengths default to class-mean separations along the direction, because
the two spaces have no common scale; each edit is also reported in units of the
global weight standard deviation, with a warning when a displacement leaves the
regime where adapters survive perturbation.

```bash
uv run python experiments/editing/edit_weights.py \
  outputs/reconstructions/x_and_x_hat_train.pt --field x \
  --identity-df path/to/identity_df.pt \
  --output-dir outputs/editing/weights \
  --attributes Bald --control-attributes Oval_Face \
  --rows 1
```

Pass `--align-with-latents` to fix the fitted population to the one the latent
run used and to verify, before fitting, that row `i` is the same adapter in both.

## Perturbation

Adds Gaussian noise scaled by the population's own spread, so the axis means the
same thing in both spaces: `z' = z + a * sigma * N(0, 1)`.

```bash
uv run python experiments/perturbation/noise_latents.py \
  outputs/latents/latents_train.pt outputs/latents/latents_val.pt \
  --output-dir outputs/perturbation/latents_075 \
  --noise-scale 0.75

uv run python experiments/perturbation/noise_weights.py \
  outputs/reconstructions/x_and_x_hat_val.pt --field x \
  --output outputs/perturbation/weights_noise.pt \
  --rows 0 1 2 3 --noise-scales 0 0.5 0.75 1.0
```

The latent script keeps each input's filename inside `--output-dir` and changes
only `z`. The weight script writes one row per selected row and noise scale, in
row-major order, with `a = 0` leaving the adapter untouched. Both report original
and perturbed statistics; `--sigma` pins the spread instead of measuring it.

## Cycles

Repeated encode/decode of the same adapters, `x_0 -> E -> z -> D -> x_1 -> ...`,
with the trained autoencoder unchanged between rounds.

```bash
uv run python experiments/cycle/cycle_latents.py \
  path/to/checkpoint.pt \
  --latents outputs/latents/latents_val.pt \
  --output-dir outputs/cycles/validation \
  --num-rounds 10 --rows 0 1 2 3
```

Writes `round_XX/x_and_x_hat.pt` per round (round zero is the identity
baseline), a combined `cycle_summary.pt` trajectory, and `cycle_config.json`.
Every round stores the same `x_0` in `x`, so cumulative metrics compare `x_0`
with `x_r` while step metrics compare consecutive `x_hat`.

For image-space drift, render every round first:

```bash
for round_dir in outputs/cycles/validation/round_*; do
  uv run --extra render python experiments/generate_images.py \
    "$round_dir/x_and_x_hat.pt" \
    --output-dir "$round_dir/images" \
    --rank 1 --fields x_hat --no-scores
done

uv run --extra evaluation python experiments/cycle/evaluate_cycles.py \
  outputs/cycles/validation --rows 0 1 2 3
```

The evaluator writes long-form scores, per-round aggregates, a paper-table CSV,
and optional figures to `<cycle-root>/evaluation/`. Pass `--no-images
--no-lpips` for weight-only evaluation.

## Tokenization figures

Two figures of how one adapter is cut into tokens. Dense chunks the flattened
adapter as a contiguous stream, so only the final token is padded; sparse
respects matrix boundaries, so most tokens carry padding.

```bash
uv run python experiments/tokenization/plot_dense_tokenization.py \
  outputs/reconstructions/x_and_x_hat_val.pt --field x --row 4 \
  --output figures/dense_tokenization.pdf

uv run python experiments/tokenization/plot_sparse_tokenization.py \
  outputs/reconstructions/x_and_x_hat_val.pt --field x --row 4 \
  --weight-dimensions path/to/weight_dimensions.pt \
  --output figures/sparse_tokenization.pdf
```

Both take any flat weight population, so no checkpoint is needed. The sparse
figure builds a real `SparseVectorTokenizer`, which needs the
`weight_dimensions.pt` that defines the matrix boundaries.

---

## Shared modules

Each experiment directory holds a module named after it — `sampling.py`,
`arithmetic.py`, `interpolation.py`, `editing.py`, `perturbation.py`,
`tokenization.py` — containing the operation shared by that experiment's latent
and weight scripts. This is what makes the pairs matched controls rather than
two implementations that happen to agree: both call the same code.

The scripts import them by plain module name (`from sampling import ...`), which
resolves because the script's own directory is on `sys.path` when you run it by
path. Run them as `python experiments/<dir>/<script>.py`, not as `-m` modules,
and keep `experiments/` itself off `PYTHONPATH`.

Infrastructure shared with the trained model — checkpoint loading, config
resolution, dataloader and tokenizer construction, decoding, and de-tokenizing —
lives in `sane.postprocessing` and is imported from there, not reimplemented.
