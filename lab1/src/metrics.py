"""Evaluation metrics, cross-domain STL-10 scoring, and synchronized inference benchmarks."""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from data import (
    CIFAR10_CLASSES,
    CIFAR10_FROG_CLASS_ID,
    CIFAR10_SHARED_9_IDS,
    prepare_inputs,
)


def synchronize(device: torch.device | str) -> None:
    """Synchronize CUDA stream for accurate GPU timing."""
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    stats: dict[str, Sequence[float]],
    device: torch.device | str,
    input_size: int = 32,
) -> dict[str, Any]:
    """Deterministically evaluate a model on a RawImages DataLoader."""
    dev = torch.device(device)
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    all_ids: list[int] = []
    all_targets: list[int] = []
    all_probs: list[np.ndarray] = []

    try:
        for raw, targets, ids in loader:
            raw = raw.to(dev, non_blocking=True)
            targets = targets.to(dev, non_blocking=True)
            logits = model(prepare_inputs(raw, stats, input_size))
            loss_sum += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            all_ids.extend(ids.tolist())
            all_targets.extend(targets.cpu().tolist())
            all_probs.append(logits.softmax(dim=1).cpu().numpy())
    finally:
        model.train(was_training)

    probabilities = np.concatenate(all_probs, axis=0)
    targets_arr = np.asarray(all_targets, dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    accuracy = float(np.mean(predictions == targets_arr))
    macro_f1 = float(f1_score(targets_arr, predictions, average="macro", zero_division=0))

    return {
        "ce": float(loss_sum / len(targets_arr)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_samples": int(len(targets_arr)),
        "ids": np.asarray(all_ids, dtype=np.int64),
        "targets": targets_arr,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def evaluate_cifar10_shared9(eval_result: dict[str, Any]) -> dict[str, Any]:
    """Compute CIFAR-10 accuracy on the 9 classes shared with STL-10 (excluding frog)."""
    targets = eval_result["targets"]
    probs = eval_result["probabilities"]
    mask = np.isin(targets, CIFAR10_SHARED_9_IDS)

    targets_9 = targets[mask]
    probs_9 = probs[mask]
    preds_10way = probs_9.argmax(axis=1)

    masked_probs = probs_9.copy()
    masked_probs[:, CIFAR10_FROG_CLASS_ID] = -np.inf
    preds_9way = masked_probs.argmax(axis=1)

    return {
        "num_samples": int(mask.sum()),
        "cifar10_shared9_10way_acc": float(np.mean(preds_10way == targets_9)),
        "cifar10_shared9_9way_masked_acc": float(np.mean(preds_9way == targets_9)),
        "cifar10_shared9_frog_leakage_rate": float(np.mean(preds_10way == CIFAR10_FROG_CLASS_ID)),
    }


@torch.inference_mode()
def evaluate_stl10_cross_domain(
    model: nn.Module,
    stl10_loader: DataLoader,
    cifar_train_stats: dict[str, Sequence[float]],
    device: torch.device | str,
    input_size: int = 32,
) -> dict[str, Any]:
    """Evaluate a frozen CIFAR-10 model on held-out STL-10 (9 shared classes).

    Reports both unmodified 10-way accuracy (with frog leakage rate) and 9-way masked accuracy.
    """
    dev = torch.device(device)
    was_training = model.training
    model.eval()
    all_targets: list[int] = []
    all_logits: list[np.ndarray] = []

    try:
        for raw, targets, _ids in stl10_loader:
            raw = raw.to(dev, non_blocking=True)
            logits = model(prepare_inputs(raw, cifar_train_stats, input_size))
            all_targets.extend(targets.tolist())
            all_logits.append(logits.cpu().numpy())
    finally:
        model.train(was_training)

    targets_arr = np.asarray(all_targets, dtype=np.int64)
    logits_arr = np.concatenate(all_logits, axis=0)
    preds_10way = logits_arr.argmax(axis=1)

    masked_logits = logits_arr.copy()
    masked_logits[:, CIFAR10_FROG_CLASS_ID] = -np.inf
    preds_9way = masked_logits.argmax(axis=1)

    per_class_9way: dict[str, float] = {}
    for cls_id in CIFAR10_SHARED_9_IDS:
        cls_mask = targets_arr == cls_id
        if cls_mask.any():
            per_class_9way[CIFAR10_CLASSES[cls_id]] = float(
                np.mean(preds_9way[cls_mask] == targets_arr[cls_mask])
            )

    return {
        "num_samples": int(len(targets_arr)),
        "stl10_10way_acc": float(np.mean(preds_10way == targets_arr)),
        "stl10_frog_leakage_rate": float(np.mean(preds_10way == CIFAR10_FROG_CLASS_ID)),
        "stl10_9way_masked_acc": float(np.mean(preds_9way == targets_arr)),
        "stl10_9way_macro_f1": float(
            f1_score(
                targets_arr,
                preds_9way,
                labels=list(CIFAR10_SHARED_9_IDS),
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_acc_9way": per_class_9way,
    }


@torch.inference_mode()
def benchmark_inference(
    model: nn.Module,
    stats: dict[str, Sequence[float]],
    device: torch.device | str,
    input_size: int = 32,
    warmup_runs: int = 10,
    measured_runs: int = 50,
    precision: str = "fp32",
) -> dict[str, Any]:
    """Measure model-only inference latency at bs=1 and throughput at bs=128."""
    dev = torch.device(device)
    was_training = model.training
    model.eval()
    device_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"

    def _time_batch(batch_size: int) -> float:
        raw = torch.rand(batch_size, 3, input_size, input_size, device=dev)
        prepared = prepare_inputs(raw, stats, input_size)
        for _ in range(warmup_runs):
            _ = model(prepared)
        synchronize(dev)
        started = time.perf_counter()
        for _ in range(measured_runs):
            _ = model(prepared)
        synchronize(dev)
        return (time.perf_counter() - started) / measured_runs

    try:
        sec_bs1 = _time_batch(1)
        sec_bs128 = _time_batch(128)
    finally:
        model.train(was_training)

    return {
        "device_name": device_name,
        "precision": precision,
        "bs1_latency_ms": float(sec_bs1 * 1000.0),
        "bs128_latency_ms_per_batch": float(sec_bs128 * 1000.0),
        "bs128_throughput_imgs_per_sec": float(128.0 / max(sec_bs128, 1e-12)),
        "warmup_runs": int(warmup_runs),
        "measured_runs": int(measured_runs),
    }
