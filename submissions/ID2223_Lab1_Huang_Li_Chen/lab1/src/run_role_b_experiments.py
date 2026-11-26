"""Role B Task 3 / Task 5, reusing the team's models and training loop."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adversarial import evaluate_attack
from corruptions import load_corruption_config, make_corruption_loaders
from data import get_or_create_split, load_cifar10_datasets, make_loaders
from io_utils import (get_environment_info, get_git_commit, load_json, register_checkpoint,
                      save_history_csv, save_json, save_numpy_archive, utc_now_iso)
from metrics import benchmark_inference, evaluate_model
from models import build_model
from role_b_analysis import (plot_attack_examples, plot_attacks, plot_baseline_diagnostics, plot_task3,
                             prediction_analysis)
from seed import set_seed
from train import ExperimentConfig, run_experiment

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "lab1/configs/role_b_experiments.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance():
    """Hashes identify uncommitted source as well as the base Git revision."""
    paths = sorted((ROOT / "lab1/src").glob("*.py")) + [
        MATRIX_PATH, ROOT / "lab1/configs/baseline.json", ROOT / "lab1/configs/split_indices.json",
    ]
    base_commit = get_git_commit(ROOT)
    archive_revision = ROOT / "lab1/role_b_source_revision.json"
    if base_commit == "unknown" and archive_revision.is_file():
        base_commit = load_json(archive_revision)["base_commit"]
    return {"base_commit": base_commit, "working_tree": "source snapshot identified by SHA256",
            "sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}}


def build_config(spec, smoke=False):
    baseline = load_json(ROOT / "lab1/configs/baseline.json")
    base = copy.deepcopy(baseline["baselines"]["alexnet_cifar"])
    options = base.pop("model_options")
    options.update(width_mult=spec["width_mult"], dropout=spec["dropout"])
    base["weight_decay"] = spec["weight_decay"]
    return ExperimentConfig(
        exp_id=f"t3_alexnet_{spec['name']}_s42", task="task3", owner="role_b",
        **base, model_options=options, seed=42, epochs=1 if smoke else baseline["epochs"],
        batch_size=baseline["batch_size"], augmentation="none", eval_train_every=1,
        notes="Role B controlled comparison; fresh initialization; smoke" if smoke else
              "Role B controlled comparison; fresh initialization; full training",
    )


def validate_training(record, config, checkpoint_path, history_path, num_train):
    """Refuse partial or differently configured runs instead of silently reusing them."""
    expected = asdict(config)
    for key, value in expected.items():
        if record["config"].get(key) != value:
            raise ValueError(f"Cached experiment differs at {key}: {config.exp_id}")
    history = pd.read_csv(history_path)
    if history.epoch.tolist() != list(range(1, config.epochs + 1)):
        raise ValueError(f"Incomplete history: {config.exp_id}")
    if record["training"]["num_trained_samples"] != config.epochs * num_train:
        raise ValueError(f"Incomplete trained-sample budget: {config.exp_id}")
    if history[["train_eval_acc", "train_eval_ce", "val_acc", "val_ce"]].isna().any().any():
        raise ValueError(f"Missing deterministic diagnostics: {config.exp_id}")
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(checkpoint_path)
    if record["role_b"]["checkpoint_sha256"] != sha256(checkpoint_path):
        raise ValueError(f"Checkpoint does not match record: {config.exp_id}")
    for relative, digest in record["role_b"]["source"]["sha256"].items():
        if Path(relative).name in ("data.py", "models.py", "train.py", "seed.py", "metrics.py",
                                   "baseline.json", "split_indices.json", "role_b_experiments.json"):
            if sha256(ROOT / relative) != digest:
                raise ValueError(f"Training input changed since cached run: {relative}")


def load_model(checkpoint_path, stats, device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = payload["config"]
    for key in ("mean", "std"):
        if not np.allclose(payload["stats"][key], stats[key], rtol=0, atol=1e-8):
            raise ValueError("Checkpoint preprocessing differs from frozen split")
    model, _ = build_model(cfg["model"], cfg["model_options"], device=device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload


def train_task3(matrix, train_ds, test_ds, split, args, device, smoke=False):
    records = []
    specs = matrix["experiments"][:1] if smoke else matrix["experiments"]
    train_ids = split["train_ids"][:128] if smoke else split["train_ids"]
    val_ids = split["val_ids"][:64] if smoke else split["val_ids"]
    for spec in specs:
        cfg = build_config(spec, smoke)
        record_path = args.results_dir / f"{cfg.exp_id}.json"
        checkpoint = args.checkpoint_root / cfg.exp_id / "best.pt"
        history = args.results_dir / "histories" / f"{cfg.exp_id}.csv"
        if record_path.exists():
            record = load_json(record_path)
            validate_training(record, cfg, checkpoint, history, len(train_ids))
            print(f"Verified existing run: {cfg.exp_id}", flush=True)
        else:
            set_seed(cfg.seed)
            loaders = make_loaders(train_ds, test_ds, train_ids, val_ids,
                                   batch_size=cfg.batch_size, seed=cfg.seed, num_workers=0)
            print(f"Training {cfg.exp_id}: {len(train_ids)} images x {cfg.epochs} epochs", flush=True)
            model, record, _ = run_experiment(
                cfg, loaders, split["stats"], device, args.results_dir,
                checkpoint_dir=args.checkpoint_root, manifest_path=None,
                evaluate_test=False, run_benchmark=False, repo_root=ROOT,
                show_progress=not args.quiet,
            )
            source = provenance()
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            payload.update(git_commit=source["base_commit"], source=source)
            torch.save(payload, checkpoint)
            record["git_commit"] = source["base_commit"]
            record["role_b"] = {
                "name": spec["name"], "run_kind": "smoke" if smoke else "full",
                "source": source, "checkpoint_sha256": sha256(checkpoint),
                "hardware_scope": "local validation; exclude from team hardware rankings"
                if "T4" not in record["environment"]["device_name"] else "T4",
            }
            record["training"]["history_file"] = str(history)
            # Local files are not claimed to be uploaded to shared Drive.
            record["checkpoint_ref"]["is_registered_in_manifest"] = False
            save_json(record, record_path)
            register_checkpoint(args.results_dir / "role_b_checkpoints_manifest.json", {
                "exp_id": cfg.exp_id, "model": cfg.model, "owner": "role_b",
                "git_commit": record["git_commit"], "source": record["role_b"]["source"],
                "checkpoint_sha256": sha256(checkpoint),
                "drive_rel_path": f"checkpoints/{cfg.exp_id}/best.pt",
                "best_val_acc": record["training"]["best_val_acc"],
                "best_epoch": record["training"]["best_epoch"], "shared_drive_verified": False,
            })
            if smoke:
                metrics, _ = evaluate_attack(model, loaders["val"], split["stats"], device, 2 / 255)
                save_json(metrics, args.results_dir / "smoke_fgsm.json")
            del model, loaders
        records.append(record)
    return records


def evaluate_task3(matrix, loaders, split, args, device):
    records = []
    # Validate the whole predeclared matrix before opening final test results.
    for spec in matrix["experiments"]:
        cfg = build_config(spec)
        record = load_json(args.results_dir / f"{cfg.exp_id}.json")
        checkpoint = args.checkpoint_root / cfg.exp_id / "best.pt"
        validate_training(record, cfg, checkpoint,
                          args.results_dir / "histories" / f"{cfg.exp_id}.csv", len(split["train_ids"]))
        records.append(record)
    for record in records:
        checkpoint = args.checkpoint_root / record["exp_id"] / "best.pt"
        model, _ = load_model(checkpoint, split["stats"], device)
        result = evaluate_model(model, loaders["test"], split["stats"], device)
        record["evaluation"].update(clean_test_acc=result["accuracy"], clean_test_ce=result["ce"],
                                     macro_f1=result["macro_f1"])
        record["inference_benchmark"] = benchmark_inference(model, split["stats"], device)
        record["role_b"]["evaluation_environment"] = get_environment_info(device)
        save_json(record, args.results_dir / f"{record['exp_id']}.json")
        print(f"Final test {record['exp_id']}: {result['accuracy']:.4f}", flush=True)
        del model
    plot_task3(records, args.results_dir)
    return records


def evaluate_task5(matrix, test_ds, loaders, split, args, device):
    plot_baseline_diagnostics(matrix["prediction_models"], ROOT / "lab1/results/histories", args.results_dir)
    corruption_path = ROOT / "lab1/configs/corruptions_t4.json"
    corrupted_loaders = make_corruption_loaders(
        test_ds, load_corruption_config(corruption_path), num_workers=0,
    )
    attack_rows, clean_rows = [], []
    for source_id in matrix["prediction_models"]:
        checkpoint = args.checkpoint_root / source_id / "best.pt"
        model, payload = load_model(checkpoint, split["stats"], device)
        if payload["exp_id"] != source_id:
            raise ValueError("Checkpoint identity does not match requested source")
        exp_id = f"t5_{source_id}"
        result = evaluate_model(model, loaders["test"], split["stats"], device)
        details = prediction_analysis(result, test_ds, args.results_dir, exp_id)
        clean_rows.append({"exp_id": exp_id, "source_exp_id": source_id,
                           "accuracy": result["accuracy"], "ce": result["ce"],
                           "macro_f1": result["macro_f1"]})
        save_numpy_archive({k: result[k] for k in ("ids", "targets", "predictions", "probabilities")},
                           args.results_dir / "predictions" / f"{exp_id}_clean.npz")
        corruption_metrics = {}
        for name, loader in corrupted_loaders.items():
            if name == "clean":
                continue
            corrupted = evaluate_model(model, loader, split["stats"], device)
            if not np.array_equal(corrupted["ids"], result["ids"]):
                raise ValueError("Corruption image IDs differ from clean evaluation")
            clean_correct = result["predictions"] == result["targets"]
            wrong = corrupted["predictions"] != corrupted["targets"]
            corruption_metrics[name] = {
                "accuracy": corrupted["accuracy"],
                "accuracy_drop": result["accuracy"] - corrupted["accuracy"],
                "clean_correct_to_wrong_rate": float((clean_correct & wrong).sum() / clean_correct.sum()),
                "num_samples": corrupted["num_samples"],
            }
        attacks = []
        for method in ("fgsm", "random_linf"):
            for epsilon in matrix["attack"]["epsilons"]:
                metrics, arrays = evaluate_attack(model, loaders["test"], split["stats"], device,
                                                  epsilon, method, matrix["attack"]["random_seed"])
                if not np.array_equal(arrays["ids"], result["ids"]) or not np.array_equal(
                        arrays["clean_predictions"], result["predictions"]):
                    raise ValueError("Attack clean predictions do not match the same checkpoint")
                attacks.append(metrics)
                attack_rows.append({"source_exp_id": source_id, **metrics})
                archive = f"{exp_id}_{method}_eps{round(epsilon * 255)}.npz"
                save_numpy_archive(arrays, args.results_dir / "predictions" / archive)
                if method == "fgsm" and epsilon == max(matrix["attack"]["epsilons"]):
                    plot_attack_examples(model, test_ds, split["stats"], arrays, epsilon, device,
                                         args.results_dir / "figures/role_b" / f"{exp_id}_fgsm.png")
                print(f"{source_id} {method} eps={epsilon:.5f}: acc={metrics['accuracy']:.4f}, "
                      f"ASR={metrics['asr']}", flush=True)
        record = {
            "exp_id": exp_id, "task": "task5", "owner": "role_b", "git_commit": provenance()["base_commit"],
            "timestamp_utc": utc_now_iso(), "environment": get_environment_info(device),
            "config": payload["config"], "source": provenance(),
            "checkpoint_ref": {"source_exp_id": source_id, "sha256": sha256(checkpoint),
                               "training_git_commit": payload.get("git_commit", "unknown"),
                               "drive_rel_path": f"checkpoints/{source_id}/best.pt"},
            "evaluation": {"clean_test_acc": result["accuracy"], "clean_test_ce": result["ce"],
                           "macro_f1": result["macro_f1"], "adversarial_metrics": attacks,
                           "corrupted_metrics": corruption_metrics, **details},
            "corruption_config_sha256": sha256(corruption_path),
        }
        save_json(record, args.results_dir / f"{exp_id}.json")
        del model
    save_history_csv(clean_rows, args.results_dir / "role_b_task5_clean.csv")
    plot_attacks(attack_rows, args.results_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "train", "evaluate", "predictions", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=ROOT / "lab1/data")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "lab1/checkpoints")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "lab1/results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    set_seed(42)
    matrix = load_json(MATRIX_PATH)
    if args.stage == "smoke":
        args.results_dir /= "smoke_role_b"
        args.checkpoint_root /= "smoke_role_b"
    train_ds, test_ds = load_cifar10_datasets(args.data_root)
    split = get_or_create_split(train_ds, ROOT / "lab1/configs/split_indices.json")
    device = torch.device(args.device)
    print(json.dumps(get_environment_info(device)), flush=True)
    if args.stage in ("smoke", "train", "all"):
        train_task3(matrix, train_ds, test_ds, split, args, device, args.stage == "smoke")
    if args.stage in ("evaluate", "predictions", "all"):
        loaders = make_loaders(train_ds, test_ds, split["train_ids"], split["val_ids"], num_workers=0)
        if args.stage != "predictions":
            evaluate_task3(matrix, loaders, split, args, device)
        evaluate_task5(matrix, test_ds, loaders, split, args, device)


if __name__ == "__main__":
    main()
