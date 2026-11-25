"""Tables and figures from Role B's measured results; no training logic."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from adversarial import fgsm
from data import CIFAR10_CLASSES, RawImages, prepare_inputs
from io_utils import save_history_csv, save_json


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_task3(records, results_dir):
    """Three controlled comparisons; both curves use deterministic eval mode."""
    results_dir = Path(results_dir)
    by_name = {r["role_b"]["name"]: r for r in records}
    groups = {
        "capacity": ["width_half", "reference"],
        "dropout": ["dropout_zero", "dropout_quarter", "reference"],
        "l2": ["l2_zero", "l2_weak", "reference"],
    }
    rows = []
    for record in records:
        cfg, tr = record["config"], record["training"]
        rows.append({
            "exp_id": record["exp_id"], "name": record["role_b"]["name"],
            "width_mult": cfg["model_options"]["width_mult"],
            "dropout": cfg["dropout"], "weight_decay": cfg["weight_decay"],
            "params": record["model_stats"]["total_params"],
            "best_val_acc": tr["best_val_acc"], "best_val_ce": tr["best_val_ce"],
            "best_epoch": tr["best_epoch"],
            "final_train_eval_acc": tr["final_train_eval_acc"],
            "final_val_acc": pd.read_csv(results_dir / "histories" / f"{record['exp_id']}.csv").iloc[-1]["val_acc"],
            "training_sec": tr["total_train_time_sec"],
            "trained_samples": tr["num_trained_samples"],
            "peak_gpu_mem_mb": tr["peak_gpu_mem_mb"],
            "device": record["environment"]["device_name"],
            "test_acc": record["evaluation"]["clean_test_acc"],
            "bs1_latency_ms": record["inference_benchmark"].get("bs1_latency_ms"),
        })
    save_history_csv(rows, results_dir / "role_b_task3_summary.csv")
    for group, names in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for index, name in enumerate(names):
            record = by_name[name]
            history = pd.read_csv(results_dir / "histories" / f"{record['exp_id']}.csv")
            for axis, metric in zip(axes, ("ce", "acc")):
                color = f"C{index}"
                axis.plot(history.epoch, history[f"train_eval_{metric}"], "--", color=color,
                          label=f"{name}: train_eval")
                axis.plot(history.epoch, history[f"val_{metric}"], color=color, label=f"{name}: val")
                axis.set(xlabel="Epoch", ylabel="Cross-entropy" if metric == "ce" else "Accuracy")
                axis.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
        fig.suptitle(f"Task 3: {group} (same training protocol)")
        fig.tight_layout()
        save_figure(fig, results_dir / "figures" / "role_b" / f"t3_{group}.png")
    return rows


def prediction_analysis(evaluation, dataset, out_dir, exp_id):
    """Report every class; pick examples by image ID rather than manual selection."""
    out_dir = Path(out_dir)
    ids, targets, predictions, probabilities = (
        evaluation[key] for key in ("ids", "targets", "predictions", "probabilities")
    )
    labels = list(range(10))
    report = classification_report(targets, predictions, labels=labels,
                                   target_names=CIFAR10_CLASSES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(targets, predictions, labels=labels)
    save_json({"classification_report": report, "confusion_matrix": matrix.tolist()},
              out_dir / f"{exp_id}_classes.json")
    save_history_csv([{"class": name, **report[name]} for name in CIFAR10_CLASSES],
                     out_dir / f"{exp_id}_classes.csv")
    normalized = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    mesh = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for i in labels:
        for j in labels:
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if normalized[i, j] > 0.55 else "black")
    ax.set(xticks=labels, yticks=labels, xticklabels=CIFAR10_CLASSES,
           yticklabels=CIFAR10_CLASSES, xlabel="Predicted class", ylabel="True class",
           title=f"{exp_id}\nConfusion matrix (row fractions)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.colorbar(mesh, ax=ax, label="Fraction within true class")
    fig.tight_layout()
    save_figure(fig, out_dir / "figures" / "role_b" / f"{exp_id}_confusion.png")

    fig, axes = plt.subplots(2, 5, figsize=(13, 5))
    selections = []
    for row, correct in enumerate((True, False)):
        positions = np.flatnonzero((predictions == targets) == correct)
        positions = positions[np.argsort(ids[positions])][:5]
        for ax in axes[row]:
            ax.axis("off")
        for ax, pos in zip(axes[row], positions):
            image_id = int(ids[pos])
            raw, _, _ = RawImages(dataset, [image_id])[0]
            ax.imshow(raw.permute(1, 2, 0).numpy())
            ax.set_title(f"ID {image_id} | true: {CIFAR10_CLASSES[targets[pos]]}\n"
                         f"pred: {CIFAR10_CLASSES[predictions[pos]]} ({probabilities[pos].max():.2f})",
                         fontsize=8)
            selections.append({"image_id": image_id, "correct": correct})
    fig.suptitle("First five correct (top) and incorrect (bottom), ascending image ID")
    fig.tight_layout()
    save_figure(fig, out_dir / "figures" / "role_b" / f"{exp_id}_predictions.png")
    return {"classification_report": report, "confusion_matrix": matrix.tolist(),
            "example_selection": selections}


def plot_baseline_diagnostics(source_ids, history_dir, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, exp_id in enumerate(source_ids):
        history = pd.read_csv(Path(history_dir) / f"{exp_id}.csv")
        for axis, metric in zip(axes, ("ce", "acc")):
            for split, style in (("train_eval", "--"), ("val", "-")):
                axis.plot(history.epoch, history[f"{split}_{metric}"], style, color=f"C{index}",
                          label=f"{exp_id}: {split}")
            axis.set(xlabel="Epoch", ylabel="Cross-entropy" if metric == "ce" else "Accuracy")
            axis.grid(alpha=0.2)
    axes[0].legend(fontsize=6)
    fig.suptitle("Task 5: original training diagnostics for the evaluated checkpoints")
    fig.tight_layout()
    save_figure(fig, Path(out_dir) / "figures/role_b/t5_training_diagnostics.png")


def plot_attack_examples(model, dataset, stats, arrays, epsilon, device, path):
    """Show first two successes and first failure among clean-correct images."""
    clean_correct = arrays["clean_predictions"] == arrays["targets"]
    successful = clean_correct & (arrays["predictions"] != arrays["targets"])
    failed = clean_correct & ~successful
    chosen = list(np.flatnonzero(successful)[:2]) + list(np.flatnonzero(failed)[:1])
    if not chosen:
        chosen = [0]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(10, 3 * len(chosen)), squeeze=False)
    was_training = model.training
    model.eval()
    try:
        for row, pos in enumerate(chosen):
            image_id = int(arrays["ids"][pos])
            raw, target, _ = RawImages(dataset, [image_id])[0]
            raw = raw.unsqueeze(0).to(device)
            adv = fgsm(model, raw, torch.tensor([target], device=device), stats, epsilon)
            delta = adv - raw
            with torch.no_grad():
                clean_p = model(prepare_inputs(raw, stats)).softmax(1)[0].cpu().numpy()
                adv_p = model(prepare_inputs(adv, stats)).softmax(1)[0].cpu().numpy()
            images = (raw, (0.5 + 8 * delta).clamp(0, 1), adv)
            titles = (
                f"ID {image_id}; true={CIFAR10_CLASSES[target]}\n"
                f"clean={CIFAR10_CLASSES[clean_p.argmax()]} ({clean_p.max():.2f})",
                f"0.5 + 8 * perturbation\nLinf={delta.abs().max().item():.5f}",
                f"FGSM epsilon={epsilon:.5f}\n"
                f"pred={CIFAR10_CLASSES[adv_p.argmax()]} ({adv_p.max():.2f})",
            )
            for ax, image, title in zip(axes[row], images, titles):
                ax.imshow(image[0].detach().cpu().permute(1, 2, 0).numpy())
                ax.set_title(title, fontsize=9)
                ax.axis("off")
    finally:
        model.train(was_training)
    fig.tight_layout()
    save_figure(fig, path)


def plot_attacks(rows, out_dir):
    out_dir = Path(out_dir)
    save_history_csv(rows, out_dir / "role_b_task5_attacks.csv")
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (exp_id, method), group in table.groupby(["source_exp_id", "method"], sort=False):
        label = f"{exp_id}: {method}"
        axes[0].plot(group.epsilon * 255, group.accuracy, marker="o", label=label)
        axes[1].plot(group.epsilon * 255, group.asr, marker="o", label=label)
    for ax, ylabel in zip(axes, ("Accuracy (all images)", "ASR (clean-correct subset)")):
        ax.set(xlabel="Raw-pixel epsilon (units of 1/255)", ylabel=ylabel, ylim=(-0.03, 1.03))
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, out_dir / "figures" / "role_b" / "t5_fgsm_vs_random.png")
