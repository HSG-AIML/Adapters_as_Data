"""Render flattened LoRA weights with the paper's frozen diffusion model.

This script consumes a reconstruction payload produced by
``experiments/reconstruct_latents.py``. It does *not* consume SANE latents
(``z``) directly: the SANE decoder and tokenizer must first turn ``z`` into the
flat ``x_hat`` adapter weights expected here.

Expected payload fields::

    {
        "x": Tensor[N, W],       # original/reference adapter (optional by mode)
        "x_hat": Tensor[N, W],   # reconstructed or manipulated adapter
        "indices": Sequence[N],  # stable sample identifiers
        "epoch": int | None,
        "split": str | None,
        "config": dict | None,
    }

The payload is loaded with ``torch.load`` and must come from a trusted source.
The rank-specific adapter config and tensor layout are read from the ``data/``
directory beside ``run.py`` unless ``--weight-dimensions`` and
``--adapter-config`` override them, so no external weights2weights checkout is
required.

Examples:

    # Render originals and reconstructions and compute paired scores.
    uv run --extra render --extra evaluation python experiments/generate_images.py \
        outputs/reconstructions/x_and_x_hat_val.pt \
        --output-dir outputs/images/reconstructions --rank 1

    # A sampled adapter has no meaningful original x; render x_hat only.
    uv run --extra render python experiments/generate_images.py \
        outputs/sampling/weights_pca.pt --output-dir outputs/images/sampling --rank 1 \
        --fields x_hat --no-scores

The base model is downloaded from Hugging Face on first use. Access therefore
requires a network connection and enough local storage for the model cache.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from sane.lora import expected_numel, flat_weights_to_state_dict, load_weight_dimensions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

RANK_ASSETS = {
    1: {
        "dimensions": DATA_DIR / "weight_dimensions.pt",
        "adapter_config": DATA_DIR / "adapter_config.json",
    },
    2: {
        "dimensions": DATA_DIR / "weight_dimensions_unified.pt",
        "adapter_config": DATA_DIR / "adapter_config_r2.json",
    },
}

DEFAULT_MODEL = "stablediffusionapi/realistic-vision-v51"
DEFAULT_PROMPT = "sks person"
DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, unfinished"
DEFAULT_SEED = 5
DEFAULT_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 3.0
DEFAULT_RESOLUTION = 512
ADAPTER_NAME = "identity1"

CSV_FIELDNAMES = [
    "epoch",
    "sample_id",
    "weight_mse",
    "weight_r2",
    "img_mse_x_vs_xhat",
    "ssim_x_vs_xhat",
    "lpips_x_vs_xhat",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reconstructions",
        type=Path,
        help="Trusted .pt payload containing flat x and/or x_hat adapter weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for rendered PNGs, scores.csv, and render_config.json.",
    )
    parser.add_argument("--rank", type=int, choices=sorted(RANK_ASSETS), required=True)
    parser.add_argument(
        "--fields",
        nargs="+",
        choices=("x", "x_hat"),
        default=("x", "x_hat"),
        help="Payload fields to render (default: x x_hat).",
    )
    parser.add_argument(
        "--rows",
        nargs="+",
        type=int,
        help="Zero-based payload rows to render (default: all rows).",
    )
    parser.add_argument(
        "--scores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute paired weight, pixel-MSE, SSIM, and LPIPS scores (default: true).",
    )
    parser.add_argument(
        "--weight-dimensions",
        type=Path,
        help="Override the rank-specific tensor-layout file read from data/.",
    )
    parser.add_argument(
        "--adapter-config",
        type=Path,
        help="Override the rank-specific PEFT adapter config read from data/.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. cuda:0 or cpu (default: auto).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_INFERENCE_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing image files instead of failing.",
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested, but CUDA is unavailable")
    return device


def load_lora_config(path: Path, lora_config_class: type) -> Any:
    """Build a PEFT LoraConfig while tolerating metadata from older PEFT versions."""

    with path.open(encoding="utf-8") as config_file:
        values = json.load(config_file)
    accepted = set(inspect.signature(lora_config_class).parameters)
    ignored = {"auto_mapping", "base_model_name_or_path", "peft_type", "revision"}
    kwargs = {
        key: value
        for key, value in values.items()
        if key in accepted and key not in ignored
    }
    return lora_config_class(**kwargs)


class LoRAImageRenderer:
    """A cached diffusion pipeline whose single LoRA adapter is replaced in memory."""

    def __init__(
        self,
        *,
        model_name: str,
        adapter_config_path: Path,
        dimensions: dict[str, Any],
        device: torch.device,
        prompt: str,
        negative_prompt: str,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
        resolution: int,
    ) -> None:
        try:
            from diffusers import DiffusionPipeline
            from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        except ImportError as error:
            raise RuntimeError(
                "Image rendering dependencies are missing. Install them with "
                "`uv sync --extra render`."
            ) from error

        if resolution <= 0 or resolution % 8 != 0:
            raise ValueError("--resolution must be a positive multiple of 8")

        self.device = device
        self.dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.dimensions = dimensions
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.resolution = resolution
        self._set_state_dict = set_peft_model_state_dict

        print(f"Loading frozen image model {model_name!r} on {device} ...")
        self.pipeline = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        lora_config = load_lora_config(adapter_config_path, LoraConfig)
        self.pipeline.unet = get_peft_model(
            self.pipeline.unet,
            lora_config,
            adapter_name=ADAPTER_NAME,
        )
        self.pipeline.to(device)
        self.pipeline.unet.eval()

    def render(self, weights: torch.Tensor) -> Any:
        state_dict = flat_weights_to_state_dict(weights, self.dimensions)
        self._set_state_dict(
            self.pipeline.unet,
            state_dict,
            adapter_name=ADAPTER_NAME,
        )

        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        in_channels = int(self.pipeline.unet.config.in_channels)
        latents = torch.randn(
            (1, in_channels, self.resolution // 8, self.resolution // 8),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        with torch.inference_mode():
            return self.pipeline(
                self.prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                height=self.resolution,
                width=self.resolution,
                latents=latents,
            ).images[0]


def weight_metrics(x: torch.Tensor, x_hat: torch.Tensor) -> tuple[float, float]:
    x = x.float().flatten()
    x_hat = x_hat.float().flatten()
    mse = torch.mean((x - x_hat) ** 2).item()
    ss_res = torch.sum((x - x_hat) ** 2).item()
    ss_tot = torch.sum((x - x.mean()) ** 2).item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mse, r2


def image_to_array(image: Any) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def image_to_lpips_tensor(image: Any, device: torch.device) -> torch.Tensor:
    array = image_to_array(image).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_image_metrics(
    image_x: Any,
    image_x_hat: Any,
    lpips_model: Any,
    device: torch.device,
) -> tuple[float, float, float]:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as error:
        raise RuntimeError(
            "Image evaluation dependencies are missing. Install them with "
            "`uv sync --extra evaluation`."
        ) from error

    array_x = image_to_array(image_x).astype(np.float32)
    array_x_hat = image_to_array(image_x_hat).astype(np.float32)
    pixel_mse = float(np.mean((array_x - array_x_hat) ** 2))
    ssim = float(
        structural_similarity(
            array_x,
            array_x_hat,
            channel_axis=2,
            data_range=255.0,
        )
    )
    with torch.inference_mode():
        lpips = float(
            lpips_model(
                image_to_lpips_tensor(image_x, device),
                image_to_lpips_tensor(image_x_hat, device),
            ).squeeze().item()
        )
    return pixel_mse, ssim, lpips


def load_lpips_model(device: torch.device) -> Any:
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError(
            "LPIPS is required for --scores. Install evaluation dependencies with "
            "`uv sync --extra evaluation`, or pass --no-scores."
        ) from error
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    return model


def _validate_payload_field(
    payload: dict[str, Any],
    field: str,
    expected_width: int,
) -> torch.Tensor:
    if field not in payload:
        raise KeyError(f"Payload does not contain requested field {field!r}")
    values = payload[field]
    if not isinstance(values, torch.Tensor) or values.ndim < 2:
        raise ValueError(f"payload[{field!r}] must be a tensor with a sample axis")
    flattened = values.reshape(values.shape[0], -1)
    if flattened.shape[1] != expected_width:
        raise ValueError(
            f"payload[{field!r}] has {flattened.shape[1]:,} weights per sample; "
            f"rank assets expect {expected_width:,}"
        )
    return flattened


def _selected_rows(rows: Sequence[int] | None, sample_count: int) -> list[int]:
    selected = list(range(sample_count)) if rows is None else list(rows)
    invalid = [row for row in selected if row < 0 or row >= sample_count]
    if invalid:
        raise IndexError(
            f"Row indices {invalid} are outside the payload's 0..{sample_count - 1} range"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("--rows contains duplicate row indices")
    return selected


def _python_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    return value


def _safe_identifier(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _image_path(output_dir: Path, sample_id: Any, field: str) -> Path:
    suffix = "x_hat" if field == "x_hat" else "x"
    return output_dir / f"sample_{_safe_identifier(sample_id)}_{suffix}.png"


def _write_manifest(
    path: Path,
    args: argparse.Namespace,
    dimensions_path: Path,
    adapter_config_path: Path,
    device: torch.device,
    rows: list[int],
) -> None:
    manifest = {
        "reconstructions": str(args.reconstructions.resolve()),
        "rank": args.rank,
        "fields": list(args.fields),
        "rows": rows,
        "scores": args.scores,
        "weight_dimensions": str(dimensions_path.resolve()),
        "adapter_config": str(adapter_config_path.resolve()),
        "device": str(device),
        "model": args.model,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "resolution": args.resolution,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    fields = list(dict.fromkeys(args.fields))
    if args.scores and set(fields) != {"x", "x_hat"}:
        raise ValueError("--scores requires rendering both --fields x x_hat")

    assets = RANK_ASSETS[args.rank]
    dimensions_path = args.weight_dimensions or assets["dimensions"]
    adapter_config_path = args.adapter_config or assets["adapter_config"]
    for path, description in (
        (args.reconstructions, "reconstruction payload"),
        (dimensions_path, "weight dimensions"),
        (adapter_config_path, "adapter config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")

    dimensions = dict(load_weight_dimensions(dimensions_path))
    flat_width = expected_numel(dimensions)
    payload = torch.load(
        args.reconstructions,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("Reconstruction payload must be a dictionary")

    weights = {
        field: _validate_payload_field(payload, field, flat_width)
        for field in fields
    }
    sample_counts = {tensor.shape[0] for tensor in weights.values()}
    if len(sample_counts) != 1:
        raise ValueError("Requested payload fields do not have the same sample count")
    sample_count = sample_counts.pop()

    indices = payload.get("indices", list(range(sample_count)))
    if len(indices) != sample_count:
        raise ValueError(
            f"payload['indices'] has {len(indices)} entries for {sample_count} samples"
        )
    rows = _selected_rows(args.rows, sample_count)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    planned_paths = [
        _image_path(output_dir, _python_scalar(indices[row]), field)
        for row in rows
        for field in fields
    ]
    existing = [path for path in planned_paths if path.exists()]
    if existing and not args.overwrite:
        preview = ", ".join(str(path) for path in existing[:3])
        raise FileExistsError(
            f"{len(existing)} output image(s) already exist ({preview}). "
            "Use --overwrite to replace them."
        )

    device = resolve_device(args.device)
    _write_manifest(
        output_dir / "render_config.json",
        args,
        dimensions_path,
        adapter_config_path,
        device,
        rows,
    )

    print(
        f"Loaded {args.reconstructions}: {sample_count} samples, "
        f"{flat_width:,} weights per sample; rendering {len(rows)} row(s)."
    )
    renderer = LoRAImageRenderer(
        model_name=args.model,
        adapter_config_path=adapter_config_path,
        dimensions=dimensions,
        device=device,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        resolution=args.resolution,
    )
    lpips_model = load_lpips_model(device) if args.scores else None

    csv_file = None
    csv_writer = None
    if args.scores:
        csv_file = (output_dir / "scores.csv").open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        csv_writer.writeheader()

    epoch = payload.get("epoch", -1)
    try:
        for position, row in enumerate(rows, start=1):
            sample_id = _python_scalar(indices[row])
            print(f"[{position}/{len(rows)}] row={row}, sample_id={sample_id}")
            images: dict[str, Any] = {}
            for field in fields:
                image = renderer.render(weights[field][row])
                destination = _image_path(output_dir, sample_id, field)
                image.save(destination)
                images[field] = image
                print(f"  saved {destination}")

            if csv_writer is not None:
                weight_mse, weight_r2 = weight_metrics(
                    weights["x"][row],
                    weights["x_hat"][row],
                )
                image_mse, ssim, lpips = compute_image_metrics(
                    images["x"],
                    images["x_hat"],
                    lpips_model,
                    device,
                )
                csv_writer.writerow(
                    {
                        "epoch": _python_scalar(epoch),
                        "sample_id": sample_id,
                        "weight_mse": round(weight_mse, 8),
                        "weight_r2": round(weight_r2, 8),
                        "img_mse_x_vs_xhat": round(image_mse, 4),
                        "ssim_x_vs_xhat": round(ssim, 6),
                        "lpips_x_vs_xhat": round(lpips, 6),
                    }
                )
                csv_file.flush()
                print(
                    f"  R2={weight_r2:.6f}, image MSE={image_mse:.4f}, "
                    f"SSIM={ssim:.4f}, LPIPS={lpips:.4f}"
                )
    finally:
        if csv_file is not None:
            csv_file.close()

    print(f"Done. Outputs written to {output_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
