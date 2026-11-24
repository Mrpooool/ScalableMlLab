"""Reusable Matplotlib plotting helpers for Lab 1 tasks and report figures."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd
import torch

from corruptions import apply_corruption
from data import CIFAR10_CLASSES, CIFAR10_SHARED_9_IDS
from io_utils import load_json

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


# ---------------------------------------------------------------------------
# Role C figures (Task 4 corruptions, Challenge 2 STL-10, Task 6 tradeoffs).
#
# Numeric figures read the saved result JSONs / CSV tables; per-image figures read
# the saved prediction NPZ archives. Pixels are reconstructed from the original
# CIFAR-10/STL-10 datasets by image ID, and CIFAR-10 corruptions are rebuilt with
# the frozen ``apply_corruption`` spec, so no model inference ever happens here.
# ---------------------------------------------------------------------------

ROLE_C_EXP_IDS: tuple[str, ...] = (
    "t6_alexnet_cifar_no_aug_s42",
    "t4_resnet18_cifar_no_aug_s42",
    "t4_resnet18_cifar_crop_flip_s42",
    "t6_resnet18_cifar_crop_flip_s0",
    "t6_resnet18_cifar_crop_flip_s123",
)

# The three seed-42 models used for the Task 4 / Challenge 2 comparison figures.
ROLE_C_SEED42_EXP_IDS: tuple[str, ...] = (
    "t6_alexnet_cifar_no_aug_s42",
    "t4_resnet18_cifar_no_aug_s42",
    "t4_resnet18_cifar_crop_flip_s42",
)

ROLE_C_RESNET_NO_AUG_S42 = "t4_resnet18_cifar_no_aug_s42"
ROLE_C_RESNET_AUG_S42 = "t4_resnet18_cifar_crop_flip_s42"

ROLE_C_MODEL_SHORT: dict[str, str] = {
    "alexnet_cifar": "AlexNet",
    "resnet18_cifar": "ResNet-18",
}
ROLE_C_AUG_SHORT: dict[str, str] = {
    "none": "no-aug",
    "crop32_pad4_hflip05": "crop+flip",
}

_MODEL_LABEL_MAX = 24


def _role_c_label(record: dict[str, Any], max_len: int = _MODEL_LABEL_MAX) -> str:
    cfg = record["config"]
    model = ROLE_C_MODEL_SHORT.get(cfg["model"], cfg["model"])
    aug = ROLE_C_AUG_SHORT.get(cfg["augmentation"], cfg["augmentation"])
    return f"{model} {aug} s{cfg['seed']}"[:max_len]


def _role_c_condition_names(corruption_config: dict[str, Any]) -> list[str]:
    return ["clean"] + [str(spec["name"]) for spec in corruption_config["corruptions"]]


def _mean_corrupted_accuracy(record: dict[str, Any], condition_names: Sequence[str]) -> float:
    corrupted = record["evaluation"]["corrupted_metrics"]
    values = [float(corrupted[name]["accuracy"]) for name in condition_names[1:]]
    return sum(values) / len(values)


def _read_history_csv(history_path: str | Path) -> list[dict[str, Any]]:
    with Path(history_path).open("r", encoding="utf-8", newline="") as fh:
        rows = []
        for row in csv.DictReader(fh):
            rows.append({key: float(value) for key, value in row.items()})
    for row in rows:
        row["epoch"] = int(row["epoch"])
    return rows


def _load_prediction_npz(predictions_dir: str | Path, exp_id: str) -> dict[str, np.ndarray]:
    path = Path(predictions_dir) / f"{exp_id}.npz"
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def _id_prediction_map(ids: np.ndarray, predictions: np.ndarray) -> dict[int, int]:
    return {int(i): int(p) for i, p in zip(np.asarray(ids), np.asarray(predictions))}


def _cifar_corrupted_hwc_uint8(
    test_dataset: Any,
    image_id: int,
    spec: dict[str, Any],
    corruption_config: dict[str, Any],
) -> np.ndarray:
    """Rebuild the corrupted pixels for one CIFAR-10 test ID with the frozen spec."""
    hwc = np.asarray(test_dataset.data[int(image_id)])
    raw = torch.from_numpy(hwc).permute(2, 0, 1).to(torch.float32).div_(255.0)
    corrupted = apply_corruption(raw, int(image_id), spec, int(corruption_config["seed"]))
    out = (corrupted.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255)
    return out.astype(np.uint8)


def _stl_hwc_uint8(stl10_dataset: Any, orig_id: int) -> np.ndarray:
    image, _stl_label = stl10_dataset[int(orig_id)]
    return np.asarray(image)


def _hide_axis(ax: Any) -> None:
    ax.axis("off")


def _finish_figure(fig: Any, target: str | Path, tight: bool = True) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def plot_role_c_learning_curves(
    histories: dict[str, list[dict[str, Any]]],
    out_path: str | Path,
) -> Path:
    """Reuse the baseline learning-curve helper for the ResNet no-aug vs crop+flip pair."""
    return plot_baseline_learning_curves(histories, out_path)


def plot_role_c_corruption_accuracy(
    records: Sequence[dict[str, Any]],
    condition_names: Sequence[str],
    out_path: str | Path,
) -> Path:
    """Grouped bars: clean + fixed corruptions accuracy for the seed-42 models."""
    conditions = list(condition_names)
    x = np.arange(len(conditions))
    width = 0.8 / max(len(records), 1)

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for idx, record in enumerate(records):
        corrupted = record["evaluation"]["corrupted_metrics"]
        values = [float(corrupted[name]["accuracy"]) * 100.0 for name in conditions]
        offset = (idx - (len(records) - 1) / 2) * width
        ax.bar(x + offset, values, width=width * 0.9, label=_role_c_label(record))

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Task 4: Clean vs Fixed Corruption Accuracy (seed 42, CIFAR-10 test)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8.5)
    return _finish_figure(fig, out_path)


def plot_role_c_per_class_drops(
    no_aug_record: dict[str, Any],
    aug_record: dict[str, Any],
    corruption_names: Sequence[str],
    class_names: Sequence[str],
    out_path: str | Path,
) -> Path:
    """One panel per corruption: per-class accuracy drop vs clean for both ResNet-18 models."""
    names = list(class_names)
    x = np.arange(len(names))
    width = 0.38
    labels = [_role_c_label(no_aug_record), _role_c_label(aug_record)]

    fig, axes = plt.subplots(1, len(corruption_names), figsize=(5.2 * len(corruption_names), 4.2))
    axes = np.atleast_1d(axes)
    for ax, condition in zip(axes, corruption_names):
        for offset_idx, record in enumerate((no_aug_record, aug_record)):
            corrupted = record["evaluation"]["corrupted_metrics"]
            clean_pc = corrupted["clean"]["per_class_accuracy"]
            cond_pc = corrupted[condition]["per_class_accuracy"]
            drops = [
                (float(clean_pc[cls]) - float(cond_pc[cls])) * 100.0
                if clean_pc.get(cls) is not None and cond_pc.get(cls) is not None
                else np.nan
                for cls in names
            ]
            ax.bar(
                x + (offset_idx - 0.5) * width,
                drops,
                width=width,
                label=labels[offset_idx],
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Per-class accuracy drop vs clean (%)")
        ax.set_title(condition)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle("Task 4: Per-Class Corruption Sensitivity (ResNet-18 no-aug vs crop+flip, seed 42)")
    return _finish_figure(fig, out_path)


def _select_flip_ids(
    npz_data: dict[str, np.ndarray],
    condition: str,
    max_ids: int,
) -> list[int]:
    """Deterministic (ascending-ID) pick of clean-correct -> corrupted-wrong image IDs."""
    ids = np.asarray(npz_data["cifar_clean_ids"])
    targets = np.asarray(npz_data["cifar_clean_targets"])
    clean_preds = np.asarray(npz_data["cifar_clean_predictions"])
    corr_preds = np.asarray(npz_data[f"cifar_{condition}_predictions"])
    flipped = (clean_preds == targets) & (corr_preds != targets)
    positions = np.flatnonzero(flipped)[:max_ids]
    return [int(ids[pos]) for pos in positions]


def plot_role_c_corruption_examples(
    no_aug_npz: dict[str, np.ndarray],
    aug_npz: dict[str, np.ndarray],
    test_dataset: Any,
    corruption_config: dict[str, Any],
    class_names: Sequence[str],
    out_path: str | Path,
    max_per_corruption: int = 4,
) -> Path:
    """Paired original/corrupted CIFAR-10 examples for IDs the no-aug model flips.

    The same image IDs are shown for the original, the corrupted version, and both
    ResNet-18 models' predictions, so the pairing is deterministic across panels.
    """
    specs = list(corruption_config["corruptions"])
    max_cols = 2 * max_per_corruption
    class_list = list(class_names)
    no_aug_maps = {
        condition: _id_prediction_map(no_aug_npz[f"cifar_{condition}_ids"], no_aug_npz[f"cifar_{condition}_predictions"])
        for condition in ("clean", *(spec["name"] for spec in specs))
    }
    aug_maps = {
        condition: _id_prediction_map(aug_npz[f"cifar_{condition}_ids"], aug_npz[f"cifar_{condition}_predictions"])
        for condition in ("clean", *(spec["name"] for spec in specs))
    }
    true_targets = _id_prediction_map(no_aug_npz["cifar_clean_ids"], no_aug_npz["cifar_clean_targets"])
    no_aug_label = "no-aug"
    aug_label = "crop+flip"

    fig, axes = plt.subplots(len(specs), max_cols, figsize=(1.9 * max_cols, 3.4 * len(specs)))
    axes = np.asarray(axes).reshape(len(specs), max_cols)
    top_fraction = 0.90  # leaves room for the suptitle; must match subplots_adjust below

    for row_idx, spec in enumerate(specs):
        condition = spec["name"]
        selected = _select_flip_ids(no_aug_npz, condition, max_per_corruption)
        for col_idx in range(max_cols):
            _hide_axis(axes[row_idx, col_idx])
        if not selected:
            row_center_y = top_fraction * (1.0 - (row_idx + 0.5) / len(specs))
            fig.text(
                0.5,
                row_center_y,
                f"{condition}: no clean-correct → corrupted-wrong IDs",
                ha="center",
                va="center",
                fontsize=9,
            )
            continue
        for pair_idx, image_id in enumerate(selected):
            orig_ax = axes[row_idx, 2 * pair_idx]
            corr_ax = axes[row_idx, 2 * pair_idx + 1]
            orig_ax.axis("on")
            corr_ax.axis("on")
            true_name = class_list[true_targets[image_id]]

            orig_ax.imshow(np.asarray(test_dataset.data[image_id]))
            orig_ax.set_title(f"ID {image_id} clean", fontsize=8)
            no_aug_clean = class_list[no_aug_maps["clean"][image_id]]
            no_aug_corr = class_list[no_aug_maps[condition][image_id]]
            aug_clean = class_list[aug_maps["clean"][image_id]]
            aug_corr = class_list[aug_maps[condition][image_id]]
            no_aug_mark = "✓" if no_aug_maps[condition][image_id] == true_targets[image_id] else "✗"
            aug_mark = "✓" if aug_maps[condition][image_id] == true_targets[image_id] else "✗"
            orig_ax.set_xlabel(
                f"{no_aug_label}: {no_aug_clean}→{no_aug_corr}{no_aug_mark}\n"
                f"{aug_label}: {aug_clean}→{aug_corr}{aug_mark}",
                fontsize=6,
            )
            orig_ax.set_xticks([])
            orig_ax.set_yticks([])

            corrupted_hwc = _cifar_corrupted_hwc_uint8(test_dataset, image_id, spec, corruption_config)
            corr_ax.imshow(corrupted_hwc)
            corr_ax.set_title(f"ID {image_id} {condition}", fontsize=8)
            corr_ax.set_xlabel(f"true: {true_name}", fontsize=6)
            corr_ax.set_xticks([])
            corr_ax.set_yticks([])

    fig.suptitle(
        "Task 4: Paired Corruption Examples (identical image IDs for original/corrupted "
        "and both ResNet-18 seed-42 models; stronger rotations, crops, or noise can make "
        "labels ambiguous even for humans)",
        fontsize=10,
    )
    fig.subplots_adjust(hspace=0.65, top=top_fraction)
    return _finish_figure(fig, out_path, tight=False)


def plot_role_c_domain_drop(
    records: Sequence[dict[str, Any]],
    class_names: Sequence[str],
    shared_ids: Sequence[int],
    out_path: str | Path,
) -> Path:
    """Challenge 2: CIFAR-10 shared9 vs STL-10 9-way accuracy, total and per-class drop."""
    labels = [_role_c_label(record) for record in records]
    shared_names = [class_names[cls_id] for cls_id in shared_ids]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

    x = np.arange(len(records))
    width = 0.36
    cifar_accs = [r["evaluation"]["challenge2_stl10"]["cifar10_test_9way_acc"] * 100.0 for r in records]
    stl_accs = [r["evaluation"]["challenge2_stl10"]["stl10_9way_masked_acc"] * 100.0 for r in records]
    axes[0].bar(x - width / 2, cifar_accs, width=width, label="CIFAR-10 shared9 (9-way)")
    axes[0].bar(x + width / 2, stl_accs, width=width, label="STL-10 (9-way masked)")
    for idx, record in enumerate(records):
        drop = record["evaluation"]["challenge2_stl10"]["domain_drop_9way"] * 100.0
        top = max(cifar_accs[idx], stl_accs[idx])
        axes[0].text(idx, top + 1.0, f"drop {drop:.1f}", ha="center", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Cross-Domain Accuracy (9 shared classes)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8.5)

    xp = np.arange(len(shared_names))
    width_p = 0.8 / max(len(records), 1)
    for idx, record in enumerate(records):
        per_class = record["evaluation"]["challenge2_stl10"]["per_class_domain_drop_9way"]
        drops = [float(per_class[cls]) * 100.0 for cls in shared_names]
        offset = (idx - (len(records) - 1) / 2) * width_p
        axes[1].bar(xp + offset, drops, width=width_p * 0.9, label=labels[idx])
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels(shared_names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Per-class domain drop (%)")
    axes[1].set_title("CIFAR-10 shared9 − STL-10 9-way per class")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8)

    fig.suptitle("Challenge 2: CIFAR-10 → STL-10 Domain Drop (seed-42 models, frozen configs)")
    return _finish_figure(fig, out_path)


def plot_role_c_stl10_confusion(
    records: Sequence[dict[str, Any]],
    class_names: Sequence[str],
    shared_ids: Sequence[int],
    predictions_dir: str | Path,
    out_path: str | Path,
) -> Path:
    """Challenge 2: shared-9-class confusion matrices for STL-10 (from prediction NPZs)."""
    shared_names = [class_names[cls_id] for cls_id in shared_ids]
    id_to_idx = {int(cls_id): idx for idx, cls_id in enumerate(shared_ids)}
    n_models = len(records)

    fig, axes = plt.subplots(1, n_models, figsize=(4.6 * n_models, 4.4))
    axes = np.atleast_1d(axes)
    for ax, record in zip(axes, records):
        npz_data = _load_prediction_npz(predictions_dir, record["exp_id"])
        targets = np.asarray(npz_data["stl10_targets"])
        preds = np.asarray(npz_data["stl10_predictions_9way"])
        counts = np.zeros((len(shared_ids), len(shared_ids)), dtype=np.int64)
        for true_id, pred_id in zip(targets, preds):
            true_id = int(true_id)
            pred_id = int(pred_id)
            if true_id in id_to_idx and pred_id in id_to_idx:
                counts[id_to_idx[true_id], id_to_idx[pred_id]] += 1

        image = ax.imshow(counts, cmap="Blues", vmin=0)
        max_count = counts.max() if counts.size else 0
        for row_idx in range(counts.shape[0]):
            for col_idx in range(counts.shape[1]):
                value = int(counts[row_idx, col_idx])
                if value == 0:
                    continue
                ax.text(
                    col_idx,
                    row_idx,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if value > 0.6 * max_count else "black",
                )
        ax.set_xticks(np.arange(len(shared_names)))
        ax.set_yticks(np.arange(len(shared_names)))
        ax.set_xticklabels(shared_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(shared_names, fontsize=7)
        ax.set_xlabel("Predicted (9-way)", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        acc = record["evaluation"]["challenge2_stl10"]["stl10_9way_masked_acc"] * 100.0
        ax.set_title(f"{_role_c_label(record)}\nSTL-10 9-way acc {acc:.1f}%", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Challenge 2: STL-10 Shared-9 Confusion Matrices (rows=true, cols=predicted)")
    return _finish_figure(fig, out_path)


def _select_stl_failure_ids(
    prediction_data: Sequence[dict[str, np.ndarray]],
    max_ids: int,
) -> list[int]:
    """Deterministic sorted union (ascending STL ID) of 9-way-wrong IDs across models."""
    wrong: set[int] = set()
    for npz_data in prediction_data:
        targets = np.asarray(npz_data["stl10_targets"])
        preds = np.asarray(npz_data["stl10_predictions_9way"])
        for image_id, ok in zip(npz_data["stl10_ids"], preds == targets):
            if not ok:
                wrong.add(int(image_id))
    return sorted(wrong)[:max_ids]


def plot_role_c_stl10_failures(
    records: Sequence[dict[str, Any]],
    stl10_dataset: Any,
    class_names: Sequence[str],
    predictions_dir: str | Path,
    out_path: str | Path,
    max_ids: int = 6,
) -> Path:
    """Challenge 2: same STL-10 image with all three seed-42 models' 9-way predictions."""
    prediction_data = [
        _load_prediction_npz(predictions_dir, record["exp_id"]) for record in records
    ]
    selected = _select_stl_failure_ids(prediction_data, max_ids)
    n_rows = 2
    n_cols = 3
    class_list = list(class_names)
    model_labels = [_role_c_label(record, max_len=26) for record in records]
    per_model_maps: list[dict[int, tuple[int, int]]] = []
    for npz_data in prediction_data:
        per_model_maps.append(
            {
                int(image_id): (int(true_id), int(pred_id))
                for image_id, true_id, pred_id in zip(
                    npz_data["stl10_ids"],
                    npz_data["stl10_targets"],
                    npz_data["stl10_predictions_9way"],
                )
            }
        )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.9 * n_rows))
    flat_axes = np.asarray(axes).reshape(-1)
    for ax in flat_axes:
        _hide_axis(ax)
    if not selected:
        flat_axes[0].axis("on")
        flat_axes[0].text(
            0.5,
            0.5,
            "all seed-42 models are correct on every STL-10 image",
            ha="center",
            va="center",
            fontsize=9,
        )
        flat_axes[0].set_xticks([])
        flat_axes[0].set_yticks([])
    for ax, image_id in zip(flat_axes, selected):
        ax.axis("on")
        ax.imshow(_stl_hwc_uint8(stl10_dataset, image_id))
        lines = []
        true_name = None
        for pred_map, model_label in zip(per_model_maps, model_labels):
            if image_id not in pred_map:
                continue
            true_id, pred_id = pred_map[image_id]
            true_name = class_list[true_id]
            mark = "✓" if pred_id == true_id else "✗"
            lines.append(f"{model_label}: {class_list[pred_id]}{mark}")
        ax.set_title(f"STL ID {image_id} | true: {true_name}", fontsize=9)
        ax.set_xlabel("\n".join(lines), fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Challenge 2: Paired STL-10 Failures (sorted union of wrong IDs across the three "
        "seed-42 models; identical images, one forward pass per model)",
        fontsize=10,
    )
    fig.subplots_adjust(hspace=0.6, top=0.90)
    return _finish_figure(fig, out_path, tight=False)


def plot_role_c_tradeoffs(
    records: Sequence[dict[str, Any]],
    condition_names: Sequence[str],
    out_path: str | Path,
) -> Path:
    """Task 6: training time / single-image latency vs clean and mean-corrupted accuracy."""
    labels = [_role_c_label(record, max_len=24) for record in records]
    clean_accs = [float(r["evaluation"]["clean_test_acc"]) * 100.0 for r in records]
    mean_corrupted = [
        _mean_corrupted_accuracy(record, condition_names) * 100.0 for record in records
    ]
    train_secs = [float(r["training"]["total_train_time_sec"]) for r in records]
    bs1_ms = [float(r["inference_benchmark"]["bs1_latency_ms"]) for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    colors = plt.get_cmap("tab10").colors
    config_handles = []
    for ax, x_values, x_label in (
        (axes[0], train_secs, "Total training time (s, 30 epochs, T4)"),
        (axes[1], bs1_ms, "bs=1 latency (ms/image, FP32)"),
    ):
        for index, (x_val, clean_val, corr_val, label) in enumerate(
            zip(x_values, clean_accs, mean_corrupted, labels)
        ):
            color = colors[index % len(colors)]
            clean_handle = ax.scatter(
                [x_val], [clean_val], marker="o", s=55, color=color, label=label
            )
            ax.scatter([x_val], [corr_val], marker="s", s=55, color=color)
            if ax is axes[0]:
                config_handles.append(clean_handle)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, linestyle="--", alpha=0.4)
    metric_handles = [
        axes[0].scatter([], [], marker="o", s=55, color="0.35", label="Clean accuracy"),
        axes[0].scatter(
            [], [], marker="s", s=55, color="0.35", label="Mean corrupted accuracy"
        ),
    ]
    axes[0].set_title("Task 6: Training Cost vs Accuracy (all 5 frozen configs)")
    axes[1].set_title("Task 6: Inference Latency vs Accuracy (all 5 frozen configs)")
    fig.legend(
        handles=[*config_handles, *metric_handles],
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    return _finish_figure(fig, out_path, tight=False)


def plot_role_c_training_duration(
    duration_csv_path: str | Path,
    exp_ids: Sequence[str],
    out_path: str | Path,
) -> Path:
    """Task 6: validation accuracy and cumulative train time at epochs 10/20/30 (seed-42)."""
    df = pd.read_csv(duration_csv_path)
    selected = df[df["exp_id"].isin(list(exp_ids))]
    labels = {
        exp_id: (
            f"{ROLE_C_MODEL_SHORT.get(row['model'], row['model'])} "
            f"{ROLE_C_AUG_SHORT.get(row['augmentation'], row['augmentation'])} "
            f"s{int(row['seed'])}"
        )
        for exp_id, row in selected.groupby("exp_id").first().iterrows()
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for exp_id in exp_ids:
        sub = selected[selected["exp_id"] == exp_id].sort_values("epoch")
        if sub.empty:
            continue
        label = labels.get(exp_id, exp_id)
        axes[0].plot(sub["epoch"], sub["val_acc"] * 100.0, marker="o", label=label)
        axes[1].plot(
            sub["epoch"],
            sub["cumulative_train_time_sec"],
            marker="o",
            label=label,
        )
    axes[0].set(xlabel="Epoch", ylabel="Validation accuracy (%)", title="Val Accuracy at Epochs 10/20/30")
    axes[1].set(xlabel="Epoch", ylabel="Cumulative train time (s)", title="Cumulative Training Cost")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8.5)
    fig.suptitle("Task 6: Training-Duration Sensitivity (seed-42 configs, Tesla T4)")
    return _finish_figure(fig, out_path)


def generate_role_c_plots(
    results_dir: str | Path,
    test_dataset: Any,
    stl10_dataset: Any,
    corruption_config: dict[str, Any],
    out_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Generate all nine Role C figures from saved JSON/CSV/NPZ artifacts.

    No model is loaded and no CIFAR-10 test or STL-10 batch is evaluated here: numeric
    figures read the five result JSONs and the saved CSV tables, per-image figures read
    the prediction NPZ archives, and pixels are reconstructed by image ID from the
    original datasets (CIFAR-10 corruptions rebuilt with the frozen spec).

    ``stl10_dataset`` must be the raw torchvision STL-10 dataset (``stl10_loader.dataset
    .dataset``), whose original IDs match the NPZ ``stl10_ids``. Returns
    ``{figure filename: saved path}``.
    """
    results_dir = Path(results_dir)
    figures_dir = Path(out_dir) if out_dir is not None else results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    condition_names = _role_c_condition_names(corruption_config)
    records = [load_json(results_dir / f"{exp_id}.json") for exp_id in ROLE_C_EXP_IDS]
    by_id = {record["exp_id"]: record for record in records}
    seed42_records = [by_id[exp_id] for exp_id in ROLE_C_SEED42_EXP_IDS]
    no_aug = by_id[ROLE_C_RESNET_NO_AUG_S42]
    aug = by_id[ROLE_C_RESNET_AUG_S42]
    class_names = list(CIFAR10_CLASSES)
    shared_ids = list(CIFAR10_SHARED_9_IDS)
    predictions_dir = results_dir / "predictions"

    paths: dict[str, Path] = {}

    histories = {
        _role_c_label(no_aug): _read_history_csv(
            results_dir / "histories" / f"{no_aug['exp_id']}.csv"
        ),
        _role_c_label(aug): _read_history_csv(
            results_dir / "histories" / f"{aug['exp_id']}.csv"
        ),
    }
    paths["t4_learning_curves.png"] = plot_role_c_learning_curves(
        histories, figures_dir / "t4_learning_curves.png"
    )
    paths["t4_corruption_accuracy.png"] = plot_role_c_corruption_accuracy(
        seed42_records, condition_names, figures_dir / "t4_corruption_accuracy.png"
    )
    paths["t4_per_class_drops.png"] = plot_role_c_per_class_drops(
        no_aug,
        aug,
        condition_names[1:],
        class_names,
        figures_dir / "t4_per_class_drops.png",
    )
    no_aug_npz = _load_prediction_npz(predictions_dir, no_aug["exp_id"])
    aug_npz = _load_prediction_npz(predictions_dir, aug["exp_id"])
    paths["t4_corruption_examples.png"] = plot_role_c_corruption_examples(
        no_aug_npz,
        aug_npz,
        test_dataset,
        corruption_config,
        class_names,
        figures_dir / "t4_corruption_examples.png",
    )
    paths["challenge2_domain_drop.png"] = plot_role_c_domain_drop(
        seed42_records,
        class_names,
        shared_ids,
        figures_dir / "challenge2_domain_drop.png",
    )
    paths["challenge2_stl10_confusion.png"] = plot_role_c_stl10_confusion(
        seed42_records,
        class_names,
        shared_ids,
        predictions_dir,
        figures_dir / "challenge2_stl10_confusion.png",
    )
    paths["challenge2_stl10_failures.png"] = plot_role_c_stl10_failures(
        seed42_records,
        stl10_dataset,
        class_names,
        predictions_dir,
        figures_dir / "challenge2_stl10_failures.png",
    )
    paths["t6_tradeoffs.png"] = plot_role_c_tradeoffs(
        records, condition_names, figures_dir / "t6_tradeoffs.png"
    )
    paths["t6_training_duration.png"] = plot_role_c_training_duration(
        results_dir / "task6_training_duration.csv",
        ROLE_C_SEED42_EXP_IDS,
        figures_dir / "t6_training_duration.png",
    )
    return paths
