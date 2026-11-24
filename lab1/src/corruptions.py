"""Deterministic per-image test-set corruptions for Role C Task 4.

All corruptions are seeded per (seed, image_id) with a fresh CPU generator, so the
same image always receives the same corruption regardless of DataLoader worker count,
batch order, or evaluation run. Outputs keep the raw [0, 1] CHW tensor semantics of
``RawImages``; size, target, and image ID are never altered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from data import RawImages

EXPECTED_CORRUPTION_NAMES: tuple[str, ...] = (
    "rotation_15deg",
    "translation_3px",
    "gaussian_noise_std_0_05",
)

KNOWN_KINDS: tuple[str, ...] = ("rotation", "translation", "gaussian_noise")

# Fields each kind must carry beyond name/kind.
_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "rotation": ("max_degrees", "fill"),
    "translation": ("max_translate_pixels", "fill"),
    "gaussian_noise": ("std",),
}

_NUM_EXPECTED_IDS = 10_000


def _validate_spec(spec: dict[str, Any]) -> None:
    if not isinstance(spec, dict):
        raise ValueError(f"corruption spec must be a dict, got {type(spec).__name__}")
    kind = spec.get("kind")
    if kind not in KNOWN_KINDS:
        raise ValueError(f"unknown corruption kind: {kind!r}")
    missing = [field for field in _REQUIRED_FIELDS[kind] if field not in spec]
    if missing:
        raise ValueError(f"corruption spec {spec.get('name')!r} missing fields: {missing}")


def load_corruption_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the frozen corruption protocol config.

    Enforces version 1, the cifar10/test fields, exactly 10,000 unique ascending
    image IDs (0..9999), and the three fixed corruption specs in order.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    if int(config.get("version", -1)) != 1:
        raise ValueError(f"unsupported corruption config version: {config.get('version')!r}")
    if config.get("dataset") != "cifar10":
        raise ValueError(f"unexpected dataset: {config.get('dataset')!r}")
    if config.get("split") != "test":
        raise ValueError(f"unexpected split: {config.get('split')!r}")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")

    image_ids = config.get("image_ids")
    if not isinstance(image_ids, list):
        raise ValueError("image_ids must be a list")
    if len(image_ids) != _NUM_EXPECTED_IDS:
        raise ValueError(f"expected {_NUM_EXPECTED_IDS} image_ids, got {len(image_ids)}")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("image_ids contain duplicates")
    if image_ids != sorted(image_ids):
        raise ValueError("image_ids must be in ascending order")
    if image_ids != list(range(_NUM_EXPECTED_IDS)):
        raise ValueError("image_ids must be exactly list(range(10000))")

    corruptions = config.get("corruptions")
    if not isinstance(corruptions, list):
        raise ValueError("corruptions must be a list")
    names = [spec.get("name") for spec in corruptions if isinstance(spec, dict)]
    if names != list(EXPECTED_CORRUPTION_NAMES):
        raise ValueError(f"unexpected corruption names/order: {names!r}")
    for spec in corruptions:
        _validate_spec(spec)

    return config


def apply_corruption(
    raw: torch.Tensor,
    image_id: int,
    spec: dict[str, Any],
    seed: int,
) -> torch.Tensor:
    """Apply a single corruption spec to one raw [0, 1] CHW tensor.

    A fresh CPU generator is seeded with ``seed + image_id`` on every call, making the
    corruption a pure function of (seed, image_id, spec, pixel values). Geometric
    transforms use bilinear interpolation and the configured fill; every output is
    clamped to [0, 1]. Shape, dtype, target, and ID semantics are unchanged.
    """
    if raw.ndim != 3:
        raise ValueError(f"expected a CHW tensor, got shape {tuple(raw.shape)}")
    _validate_spec(spec)
    generator = torch.Generator().manual_seed(int(seed) + int(image_id))

    kind = spec["kind"]
    if kind == "rotation":
        max_degrees = float(spec["max_degrees"])
        angle = (float(torch.rand(1, generator=generator).item()) * 2.0 - 1.0) * max_degrees
        out = TF.affine(
            raw,
            angle=angle,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=float(spec["fill"]),
        )
        out = out.clamp_(0.0, 1.0)
    elif kind == "translation":
        max_t = int(spec["max_translate_pixels"])
        dx = int(torch.randint(-max_t, max_t + 1, (1,), generator=generator).item())
        dy = int(torch.randint(-max_t, max_t + 1, (1,), generator=generator).item())
        out = TF.affine(
            raw,
            angle=0.0,
            translate=[dx, dy],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=float(spec["fill"]),
        )
        out = out.clamp_(0.0, 1.0)
    else:  # gaussian_noise
        std = float(spec["std"])
        noise = torch.randn(raw.shape, generator=generator, dtype=raw.dtype)
        out = (raw + noise * std).clamp_(0.0, 1.0)

    return out.contiguous()


class CorruptedImages(Dataset):
    """RawImages over a fixed ID list with one corruption spec applied per sample.

    Returns (corrupted_tensor, target, image_id) in the same (target, ID) order as the
    underlying ``RawImages``, so corrupted and clean loaders align sample-by-sample.
    """

    def __init__(
        self,
        test_dataset: Dataset,
        image_ids: Sequence[int],
        spec: dict[str, Any],
        seed: int,
    ) -> None:
        _validate_spec(spec)
        self.base = RawImages(test_dataset, ids=image_ids)
        self.spec = spec
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        raw, target, image_id = self.base[idx]
        corrupted = apply_corruption(raw, image_id, self.spec, self.seed)
        return corrupted, target, image_id


def make_corruption_loaders(
    test_dataset: Dataset,
    config: dict[str, Any],
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool | None = None,
) -> dict[str, DataLoader]:
    """Build clean + one loader per corruption spec, all deterministic and ID-aligned.

    Every loader iterates the config's image_ids in the same ascending order with
    ``shuffle=False``; corruption randomness comes from per-sample generators, so
    num_workers=0 and num_workers=2 produce bit-identical inputs.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    ids = [int(i) for i in config["image_ids"]]
    seed = int(config["seed"])
    common_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "shuffle": False,
    }

    loaders: dict[str, DataLoader] = {
        "clean": DataLoader(RawImages(test_dataset, ids=ids), **common_kwargs)
    }
    for spec in config["corruptions"]:
        loaders[spec["name"]] = DataLoader(
            CorruptedImages(test_dataset, ids, spec, seed),
            **common_kwargs,
        )
    return loaders
