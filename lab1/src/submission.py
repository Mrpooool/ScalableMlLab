"""Audit and display the saved Lab 1 experiments without retraining them."""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from IPython.display import Code, Image, display


def audit_results(root):
    """Check the split, all full training histories, and paired prediction metrics."""
    root = Path(root)
    results = root / "lab1/results"
    split = json.loads((root / "lab1/configs/split_indices.json").read_text())
    train_ids, val_ids = split["train_ids"], split["val_ids"]
    assert len(train_ids) == 45000 and len(val_ids) == 5000
    assert len(set(train_ids) | set(val_ids)) == 50000
    assert not set(train_ids) & set(val_ids)
    manifest = json.loads((results / "checkpoints_manifest.json").read_text())
    rows = []
    for entry in manifest["checkpoints"]:
        exp_id = entry["exp_id"]
        record = json.loads((results / f"{exp_id}.json").read_text())
        history = pd.read_csv(results / "histories" / f"{exp_id}.csv")
        assert history["epoch"].tolist() == list(range(1, 31)), exp_id
        assert record["training"]["num_trained_samples"] == 1350000, exp_id
        best = history.sort_values(["val_acc", "val_ce"], ascending=[False, True]).iloc[0]
        assert int(best["epoch"]) == record["training"]["best_epoch"], exp_id
        assert np.isclose(best["val_acc"], record["training"]["best_val_acc"]), exp_id
        rows.append({"exp_id": exp_id, "epochs": len(history),
                     "training_device": record["environment"]["device_name"],
                     "best_val_acc": float(best["val_acc"]),
                     "test_acc": record["evaluation"]["clean_test_acc"]})
    assert len(rows) == 13
    checked = 0
    for path in (results / "predictions").glob("t[46]_*.npz"):
        record = json.loads((results / f"{path.stem}.json").read_text())
        with np.load(path) as arrays:
            for condition, metrics in record["evaluation"]["corrupted_metrics"].items():
                prefix = f"cifar_{condition}"
                assert np.array_equal(arrays[prefix + "_ids"], arrays["cifar_clean_ids"])
                assert len(arrays[prefix + "_ids"]) == 10000
                accuracy = np.mean(arrays[prefix + "_targets"] == arrays[prefix + "_predictions"])
                assert np.isclose(accuracy, metrics["accuracy"]), (path, condition)
                checked += 1
            assert len(arrays["stl10_ids"]) == 7200
            accuracy = np.mean(arrays["stl10_targets"] == arrays["stl10_predictions_9way"])
            assert np.isclose(accuracy, record["evaluation"]["challenge2_stl10"]["stl10_9way_masked_acc"])
            checked += 1
    attacks = pd.read_csv(results / "role_b_task5_attacks.csv")
    for source_id in attacks["source_exp_id"].unique():
        with np.load(results / "predictions" / f"t5_{source_id}_clean.npz") as clean:
            for row in attacks[attacks["source_exp_id"] == source_id].to_dict("records"):
                budget = round(row["epsilon"] * 255)
                path = results / "predictions" / f"t5_{source_id}_{row['method']}_eps{budget}.npz"
                with np.load(path) as attacked:
                    assert np.array_equal(clean["ids"], attacked["ids"])
                    assert np.array_equal(clean["targets"], attacked["targets"])
                    correct = clean["predictions"] == clean["targets"]
                    wrong = attacked["predictions"] != attacked["targets"]
                    assert len(correct) == 10000
                    assert np.isclose(1 - wrong.mean(), row["accuracy"])
                    assert np.isclose((correct & wrong).sum() / correct.sum(), row["asr"])
                    assert row["max_observed_linf"] <= row["epsilon"] + 1e-6
                    checked += 1
    print("Display/audit runtime:", platform.platform(), "| Python", platform.python_version(),
          "| torch", torch.__version__, "| GPU used: none")
    print(f"PASS: disjoint 45k/5k split; {len(rows)} complete 30-epoch histories; "
          f"{checked} prediction conditions agree with saved metrics.")
    print("Saved measurements retain their original training and inference hardware.")
    display(pd.DataFrame(rows))


def show_task(root, task):
    """Display each task's measured tables and figures from the submitted artifacts."""
    root = Path(root)
    results = root / "lab1/results"
    tables = {
        "1": [], "2": ["summary.csv"], "3": ["role_b_task3_summary.csv"],
        "4": ["role_c_summary.csv"],
        "5": ["role_b_task5_clean.csv", "role_b_task5_attacks.csv"],
        "6": ["role_c_summary.csv", "task6_seed_sensitivity.csv", "task6_training_duration.csv"],
        "challenge2": ["role_c_summary.csv"],
    }
    figures = {
        "1": ["t1_class_distribution.png", "t1_sample_grid.png"],
        "2": ["t2_learning_curves.png", "t2_baseline_comparison.png"],
        "3": [f"role_b/t3_{name}.png" for name in ["capacity", "dropout", "l2"]],
        "4": ["t4_learning_curves.png", "t4_corruption_accuracy.png", "t4_per_class_drops.png", "t4_corruption_examples.png"],
        "5": ["role_b/t5_training_diagnostics.png", "role_b/t5_fgsm_vs_random.png"],
        "6": ["t6_tradeoffs.png", "t6_training_duration.png"],
        "challenge2": ["challenge2_domain_drop.png", "challenge2_stl10_confusion.png", "challenge2_stl10_failures.png"],
    }
    if task == "1":
        summary = json.loads((results / "task1_data_summary.json").read_text())
        print("Image shape:", summary["image_shape_hwc"], "| dtype:", summary["dtype"])
        print("Split counts:", summary["counts"])
        display(pd.DataFrame(summary["train_only_stats"], index=["red", "green", "blue"]))
        display(pd.DataFrame(summary["class_distribution"]).T)
    if task == "2":
        from models import build_model, count_parameters, trace_shapes
        profiles = json.loads((results / "task2_architecture_profiles.json").read_text())
        torch.set_num_threads(2)
        for name, profile in profiles.items():
            model, size = build_model(name, device="cpu")
            assert count_parameters(model) == profile["parameters"]
            shapes = trace_shapes(model, size, "cpu")
            assert shapes == profile["stage_shapes"]
            print(name, count_parameters(model), "| fresh CPU architecture check passed")
            display(pd.DataFrame(shapes))
    if task == "5":
        ids = json.loads((root / "lab1/configs/role_b_experiments.json").read_text())["prediction_models"]
        for source_id in ids:
            tables[task].append(f"t5_{source_id}_classes.csv")
            figures[task].extend(f"role_b/t5_{source_id}_{suffix}.png"
                                 for suffix in ["confusion", "predictions", "fgsm"])
    for name in tables[task]:
        print(name)
        display(pd.read_csv(results / name))
    for name in figures[task]:
        display(Image(filename=str(results / "figures" / name)))


def show_implementation(root):
    """Include complete Python implementation and frozen settings in notebook outputs."""
    root = Path(root)
    paths = [*sorted((root / "lab1/src").glob("*.py")),
             *sorted((root / "lab1/configs").glob("*.json"))]
    for path in paths:
        raw = path.read_bytes()
        print(path.relative_to(root), "| SHA256", hashlib.sha256(raw).hexdigest())
        if path.name == "split_indices.json":
            print("Full 50,000-ID split is included in the source archive; audited above.")
        else:
            display(Code(raw.decode("utf-8"), language="python" if path.suffix == ".py" else "json"))
