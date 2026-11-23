"""Dataset loading, fixed train/val split, preprocessing, and DataLoader builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import functional as TF

from seed import make_generator, worker_init_fn

CIFAR10_MIRROR_URL = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
CIFAR10_FILENAME = "cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"

SPLIT_SEED = 2223
VAL_RATIO = 0.1

CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Challenge 2: exact label mapping from STL-10 (excluding 7: monkey) to CIFAR-10 indices.
STL10_TO_CIFAR10: dict[int, int] = {
    0: 0,  # airplane -> airplane
    1: 2,  # bird -> bird
    2: 1,  # car -> automobile
    3: 3,  # cat -> cat
    4: 4,  # deer -> deer
    5: 5,  # dog -> dog
    6: 7,  # horse -> horse
    8: 8,  # ship -> ship
    9: 9,  # truck -> truck
}

CIFAR10_SHARED_9_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 7, 8, 9)
CIFAR10_FROG_CLASS_ID = 6


def load_cifar10_datasets(
    data_root: str | Path,
) -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Load CIFAR-10 train and test splits, preferring the fast Brainchip mirror."""
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        train_ds = datasets.CIFAR10(root, train=True, download=False)
        test_ds = datasets.CIFAR10(root, train=False, download=False)
        return train_ds, test_ds
    except RuntimeError:
        pass

    try:
        download_and_extract_archive(
            CIFAR10_MIRROR_URL,
            download_root=str(root),
            filename=CIFAR10_FILENAME,
            md5=CIFAR10_MD5,
        )
        train_ds = datasets.CIFAR10(root, train=True, download=False)
        test_ds = datasets.CIFAR10(root, train=False, download=False)
        return train_ds, test_ds
    except Exception:
        train_ds = datasets.CIFAR10(root, train=True, download=True)
        test_ds = datasets.CIFAR10(root, train=False, download=True)
        return train_ds, test_ds


def compute_train_stats(
    train_dataset: datasets.CIFAR10,
    train_ids: Sequence[int],
    batch_size: int = 256,
) -> dict[str, list[float]]:
    """Compute per-channel RGB mean and std on training split images only."""
    ids = np.asarray(train_ids, dtype=np.int64)
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        pixels = train_dataset.data[batch_ids].astype(np.float64) / 255.0
        channel_sum += pixels.sum(axis=(0, 1, 2))
        channel_square_sum += np.square(pixels).sum(axis=(0, 1, 2))
        pixel_count += int(np.prod(pixels.shape[:3]))

    mean = channel_sum / pixel_count
    std = np.sqrt(np.maximum(channel_square_sum / pixel_count - mean**2, 1e-12))
    return {"mean": mean.tolist(), "std": std.tolist()}


def get_or_create_split(
    train_dataset: datasets.CIFAR10,
    split_path: str | Path,
    val_ratio: float = VAL_RATIO,
    split_seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Load the frozen stratified train/val split or create and persist it once."""
    path = Path(split_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        train_ids = payload["train_ids"]
        val_ids = payload["val_ids"]
        assert set(train_ids).isdisjoint(val_ids), "Train and val splits overlap!"
        assert len(train_ids) + len(val_ids) == len(train_dataset), "Split size mismatch!"
        return payload

    labels = np.asarray(train_dataset.targets)
    train_ids, val_ids = train_test_split(
        np.arange(len(train_dataset)),
        test_size=val_ratio,
        stratify=labels,
        random_state=split_seed,
    )
    train_ids = np.sort(train_ids).tolist()
    val_ids = np.sort(val_ids).tolist()
    assert set(train_ids).isdisjoint(val_ids)

    stats = compute_train_stats(train_dataset, train_ids)
    payload = {
        "split_seed": int(split_seed),
        "val_ratio": float(val_ratio),
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "stats": stats,
        "train_ids": train_ids,
        "val_ids": val_ids,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    tmp_path.replace(path)
    return payload


class RawImages(Dataset):
    """Dataset wrapper returning raw [0, 1] CHW tensor, integer label, and original image ID."""

    def __init__(
        self,
        dataset: Dataset,
        ids: Sequence[int] | None = None,
        pil_transform: Callable | None = None,
        tensor_transform: Callable | None = None,
    ) -> None:
        self.dataset = dataset
        self.ids = list(range(len(dataset))) if ids is None else [int(i) for i in ids]
        self.pil_transform = pil_transform
        self.tensor_transform = tensor_transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        image_id = self.ids[idx]
        image, target = self.dataset[image_id]
        if self.pil_transform is not None:
            image = self.pil_transform(image)
        tensor = TF.to_tensor(image) if not isinstance(image, torch.Tensor) else image
        if self.tensor_transform is not None:
            tensor = self.tensor_transform(tensor)
        return tensor, int(target), image_id


def prepare_inputs(
    raw: torch.Tensor,
    stats: dict[str, Sequence[float]],
    size: int = 32,
) -> torch.Tensor:
    """Differentiably resize (if needed) and normalize raw [0, 1] image tensors."""
    if raw.shape[-2:] != (size, size):
        resized = nn.functional.interpolate(
            raw,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    else:
        resized = raw
    mean = resized.new_tensor(stats["mean"])[None, :, None, None]
    std = resized.new_tensor(stats["std"])[None, :, None, None]
    return (resized - mean) / std


def make_loaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    batch_size: int = 128,
    seed: int = 42,
    train_pil_transform: Callable | None = None,
    train_tensor_transform: Callable | None = None,
    num_workers: int = 2,
    pin_memory: bool | None = None,
) -> dict[str, DataLoader]:
    """Create train, deterministic train_eval, val, and test DataLoaders."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    common_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        common_kwargs["worker_init_fn"] = worker_init_fn

    return {
        "train": DataLoader(
            RawImages(
                train_dataset,
                train_ids,
                pil_transform=train_pil_transform,
                tensor_transform=train_tensor_transform,
            ),
            shuffle=True,
            generator=make_generator(seed),
            **common_kwargs,
        ),
        "train_eval": DataLoader(
            RawImages(train_dataset, train_ids),
            shuffle=False,
            **common_kwargs,
        ),
        "val": DataLoader(
            RawImages(train_dataset, val_ids),
            shuffle=False,
            **common_kwargs,
        ),
        "test": DataLoader(
            RawImages(test_dataset),
            shuffle=False,
            **common_kwargs,
        ),
    }


class STL10Shared9Dataset(Dataset):
    """Held-out STL-10 test set filtered to the 9 classes shared with CIFAR-10."""

    def __init__(self, stl10_dataset: datasets.STL10) -> None:
        self.dataset = stl10_dataset
        labels = np.asarray(stl10_dataset.labels, dtype=np.int64)
        self.valid_indices = [
            int(idx) for idx, lbl in enumerate(labels) if int(lbl) in STL10_TO_CIFAR10
        ]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        orig_id = self.valid_indices[idx]
        image, stl_label = self.dataset[orig_id]
        cifar_label = STL10_TO_CIFAR10[int(stl_label)]
        tensor = TF.to_tensor(image) if not isinstance(image, torch.Tensor) else image
        return tensor, cifar_label, orig_id


def make_stl10_test_loader(
    data_root: str | Path,
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Download/load STL-10 test split filtered and mapped to CIFAR-10's 9 shared classes."""
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    stl_test = datasets.STL10(root=str(root), split="test", download=True)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        STL10Shared9Dataset(stl_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
