"""Reusable Matplotlib plotting helpers for Lab 1 tasks and report figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_class_distribution(
    train_targets: Sequence[int],
    val_targets: Sequence[int],
    test_targets: Sequence[int],
    class_names: Sequence[str],
    out_path: str | Path,
) -> Path:
    """Save a grouped bar chart showing CIFAR-10 train / val / test class counts."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)
    tr_counts = np.bincount(np.asarray(train_targets), minlength=n_classes)
    va_counts = np.bincount(np.asarray(val_targets), minlength=n_classes)
    te_counts = np.bincount(np.asarray(test_targets), minlength=n_classes)

    x = np.arange(n_classes)
    width = 0.27

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - width, tr_counts, width=width, label=f"Train ({len(train_targets):,})")
    ax.bar(x, va_counts, width=width, label=f"Val ({len(val_targets):,})")
    ax.bar(x + width, te_counts, width=width, label=f"Test ({len(test_targets):,})")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Number of images")
    ax.set_title("CIFAR-10 Stratified Split Class Distribution (SPLIT_SEED=2223)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def plot_sample_grid(
    images_hwc_uint8: np.ndarray,
    targets: Sequence[int],
    ids: Sequence[int],
    class_names: Sequence[str],
    out_path: str | Path,
    per_class: int = 3,
) -> Path:
    """Save a representative grid of CIFAR-10 images (one row per class or 2x5)."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(targets)
    id_arr = np.asarray(ids)
    n_classes = len(class_names)

    fig, axes = plt.subplots(n_classes, per_class, figsize=(per_class * 2.0, n_classes * 1.35))
    for cls_idx in range(n_classes):
        matches = np.flatnonzero(labels == cls_idx)[:per_class]
        for col_idx, pos in enumerate(matches):
            ax = axes[cls_idx, col_idx] if per_class > 1 else axes[cls_idx]
            ax.imshow(images_hwc_uint8[pos])
            if col_idx == 0:
                ax.set_ylabel(class_names[cls_idx], rotation=0, labelpad=42, va="center", fontsize=9)
            ax.set_title(f"ID {int(id_arr[pos])}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Representative CIFAR-10 Training Images (32x32 RGB)", y=0.995, fontsize=11)
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def plot_baseline_learning_curves(
    histories: dict[str, list[dict[str, Any]]],
    out_path: str | Path,
) -> Path:
    """Plot deterministic train_eval vs val loss and accuracy curves for multiple models."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (label, rows) in enumerate(histories.items()):
        df = pd.DataFrame(rows)
        color = colors[idx % len(colors)]
        axes[0].plot(
            df["epoch"],
            df["train_eval_ce"],
            linestyle="--",
            color=color,
            alpha=0.85,
            label=f"{label} (train_eval)",
        )
        axes[0].plot(
            df["epoch"],
            df["val_ce"],
            linestyle="-",
            color=color,
            linewidth=2.0,
            label=f"{label} (val)",
        )
        axes[1].plot(
            df["epoch"],
            df["train_eval_acc"] * 100.0,
            linestyle="--",
            color=color,
            alpha=0.85,
            label=f"{label} (train_eval)",
        )
        axes[1].plot(
            df["epoch"],
            df["val_acc"] * 100.0,
            linestyle="-",
            color=color,
            linewidth=2.0,
            label=f"{label} (val)",
        )

    axes[0].set(xlabel="Epoch", ylabel="Cross-Entropy Loss", title="Loss Curves (Deterministic Eval)")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy Curves (Deterministic Eval)")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def plot_baseline_comparison(
    records: Sequence[dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """Save a 3-panel summary bar chart comparing accuracy, params/memory, and speed."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    names = [r["config"]["model"] for r in records]
    val_accs = [r["training"]["best_val_acc"] * 100.0 for r in records]
    test_accs = [(r["evaluation"]["clean_test_acc"] or 0.0) * 100.0 for r in records]
    params_m = [r["model_stats"]["total_params"] / 1e6 for r in records]
    train_secs = [r["training"]["total_train_time_sec"] for r in records]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    w = 0.34
    axes[0].bar(x - w / 2, val_accs, width=w, label="Best Val Acc")
    axes[0].bar(x + w / 2, test_accs, width=w, label="Clean Test Acc")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Validation vs Test Accuracy")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8.5)

    axes[1].bar(x, params_m, width=0.45, color="#4C72B0")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("Parameters (Millions)")
    axes[1].set_title("Model Parameter Count")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    axes[2].bar(x, train_secs, width=0.45, color="#DD8452")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names)
    axes[2].set_ylabel("Total Train Time (s)")
    axes[2].set_title("30-Epoch Training Cost")
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target
