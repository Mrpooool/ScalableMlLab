"""Role C runner: frozen T4 experiment matrix for Task 4, Challenge 2, and Task 6.

Stages:
  smoke    - 1 epoch on 512 train / 128 val IDs with an isolated smoke results/checkpoint
             directory; never iterates CIFAR-10 test and never loads STL-10.
  train    - train the five frozen EXPERIMENT_SPECS rows with resume validation
             (evaluate_test=False, run_benchmark=False; no test/STL access).
  evaluate - requires all five valid training records before any CIFAR-10 test or STL-10
             access, then runs the unified evaluation: fixed corruptions (Task 4),
             STL-10 cross-domain (Challenge 2), and inference benchmark (Task 6),
             updating the five result JSONs and writing the Role C CSV tables.
  all      - train followed by evaluate.

All stages assert the CUDA device is exactly a Tesla T4; results from any other
hardware are never substituted into the T4 tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torchvision import transforms

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from corruptions import (  # noqa: E402
    EXPECTED_CORRUPTION_NAMES,
    load_corruption_config,
    make_corruption_loaders,
)
from data import (  # noqa: E402
    get_or_create_split,
    load_cifar10_datasets,
    make_loaders,
    make_stl10_test_loader,
)
from io_utils import (  # noqa: E402
    load_checkpoint_weights,
    load_json,
    save_history_csv,
    save_json,
    save_numpy_archive,
)
from metrics import (  # noqa: E402
    benchmark_inference,
    build_cross_domain_summary,
    evaluate_cifar10_shared9,
    evaluate_model,
    evaluate_stl10_cross_domain,
    synchronize,
)
from models import build_model  # noqa: E402
from plots import generate_role_c_plots  # noqa: E402
from train import ExperimentConfig, run_experiment  # noqa: E402

EXPECTED_DEVICE_NAME = "Tesla T4"
EVAL_DEVICE_NAME = "NVIDIA RTX 2000 Ada Generation"
EXPECTED_PRECISION = "fp32"
EXPECTED_OWNER = "role_c"
# The shared Drive results/ directory rejects new files; keep Role C outputs independent.
RESULTS_DIRNAME = "results_role_c"
# 45,000 frozen train IDs x 30 epochs.
EXPECTED_TRAINED_SAMPLES = 1_350_000

AUGMENTATION_NONE = "none"
AUGMENTATION_CROP_FLIP = "crop32_pad4_hflip05"

SMOKE_EXP_ID = "smoke_role_c_resnet18_crop_flip_s42"
SMOKE_EPOCHS = 1
SMOKE_TRAIN_SAMPLES = 512
SMOKE_VAL_SAMPLES = 128

# Frozen Role C training matrix (Task 4 / Task 6); hyperparameters come from baseline.json.
EXPERIMENT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "exp_id": "t6_alexnet_cifar_no_aug_s42",
        "task": "task6",
        "model": "alexnet_cifar",
        "seed": 42,
        "augmentation": AUGMENTATION_NONE,
    },
    {
        "exp_id": "t4_resnet18_cifar_no_aug_s42",
        "task": "task4",
        "model": "resnet18_cifar",
        "seed": 42,
        "augmentation": AUGMENTATION_NONE,
    },
    {
        "exp_id": "t4_resnet18_cifar_crop_flip_s42",
        "task": "task4",
        "model": "resnet18_cifar",
        "seed": 42,
        "augmentation": AUGMENTATION_CROP_FLIP,
    },
    {
        "exp_id": "t6_resnet18_cifar_crop_flip_s0",
        "task": "task6",
        "model": "resnet18_cifar",
        "seed": 0,
        "augmentation": AUGMENTATION_CROP_FLIP,
    },
    {
        "exp_id": "t6_resnet18_cifar_crop_flip_s123",
        "task": "task6",
        "model": "resnet18_cifar",
        "seed": 123,
        "augmentation": AUGMENTATION_CROP_FLIP,
    },
)

# Task 6 seed-sensitivity group: augmented ResNet-18 at seeds 0/42/123.
SEED_SENSITIVITY_EXP_IDS: tuple[str, ...] = (
    "t6_resnet18_cifar_crop_flip_s0",
    "t4_resnet18_cifar_crop_flip_s42",
    "t6_resnet18_cifar_crop_flip_s123",
)

SEED_SENSITIVITY_METRICS: tuple[str, ...] = (
    "best_val_acc",
    "clean_test_acc",
    "rotation_15deg_acc",
    "translation_3px_acc",
    "gaussian_noise_std_0_05_acc",
    "mean_corrupted_acc",
    "stl10_9way_masked_acc",
    "domain_drop_9way",
    "total_train_time_sec",
)

# Epochs sampled for the training-duration sensitivity table (validation only).
TRAINING_DURATION_EPOCHS: tuple[int, ...] = (10, 20, 30)

# Role A same-architecture RTX 5070 Ti records (REPO/lab1/results) paired with the
# corresponding Role C no-augmentation T4 rows for the cross-hardware reference table.
HARDWARE_COMPARISON_PAIRS: tuple[tuple[str, str], ...] = (
    ("t2_alexnet_cifar_baseline_s42", "t6_alexnet_cifar_no_aug_s42"),
    ("t2_resnet18_cifar_baseline_s42", "t4_resnet18_cifar_no_aug_s42"),
)

HARDWARE_COMPARISON_NOTE = (
    "different hardware (RTX 5070 Ti vs Tesla T4); reference only, "
    "not a same-device performance comparison"
)


def compare_corrupted_to_clean(
    clean_targets: np.ndarray,
    clean_predictions: np.ndarray,
    corrupted_predictions: np.ndarray,
) -> dict[str, Any]:
    """Compare one corruption condition against clean predictions aligned by image ID.

    All arrays must refer to the same images in the same order. The
    clean-correct->wrong rate is relative to the number of clean-correct samples and is
    None when the model got nothing right on the clean split.
    """
    targets = np.asarray(clean_targets)
    clean_preds = np.asarray(clean_predictions)
    corrupted_preds = np.asarray(corrupted_predictions)
    if not (targets.shape == clean_preds.shape == corrupted_preds.shape):
        raise ValueError(
            "clean targets/predictions and corrupted predictions must share one shape, "
            f"got {targets.shape}, {clean_preds.shape}, {corrupted_preds.shape}"
        )
    clean_correct = clean_preds == targets
    num_clean_correct = int(clean_correct.sum())
    clean_correct_to_wrong = int(
        np.count_nonzero(clean_correct & (corrupted_preds != targets))
    )
    return {
        "prediction_flip_rate_vs_clean": float(np.mean(corrupted_preds != clean_preds)),
        "clean_correct_to_wrong_count": clean_correct_to_wrong,
        "clean_correct_to_wrong_rate": (
            float(clean_correct_to_wrong / num_clean_correct)
            if num_clean_correct > 0
            else None
        ),
    }


def summarize_seed_metrics(values: Sequence[float]) -> dict[str, float]:
    """Mean / sample std (ddof=1) / min / max over the seed-sensitivity group."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot summarize an empty metric list")
    return {
        "mean": float(arr.mean()),
        "sample_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _mean_corruption_accuracy(corrupted_metrics: dict[str, Any]) -> float:
    """Exact arithmetic mean of the three fixed corruption accuracies."""
    values = [float(corrupted_metrics[name]["accuracy"]) for name in EXPECTED_CORRUPTION_NAMES]
    return sum(values) / len(values)


def build_role_c_summary_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flat one-row-per-experiment table for results/role_c_summary.csv."""
    rows: list[dict[str, Any]] = []
    for record in records:
        cfg = record["config"]
        training = record["training"]
        evaluation = record["evaluation"]
        benchmark = record["inference_benchmark"]
        corrupted = evaluation["corrupted_metrics"]
        challenge2 = evaluation["challenge2_stl10"]
        row: dict[str, Any] = {
            "exp_id": record["exp_id"],
            "task": record["task"],
            "model": cfg["model"],
            "seed": cfg["seed"],
            "augmentation": cfg["augmentation"],
            "device_name": record["environment"]["device_name"],
            "training_device_name": record["environment"]["device_name"],
            "inference_device_name": benchmark["device_name"],
            "total_params": record["model_stats"]["total_params"],
            "best_epoch": training["best_epoch"],
            "best_val_acc": training["best_val_acc"],
            "clean_test_acc": evaluation["clean_test_acc"],
        }
        for name in EXPECTED_CORRUPTION_NAMES:
            row[f"{name}_acc"] = corrupted[name]["accuracy"]
        row["mean_corrupted_acc"] = _mean_corruption_accuracy(corrupted)
        row.update(
            {
                "stl10_10way_acc": challenge2["stl10_10way_acc"],
                "stl10_9way_masked_acc": challenge2["stl10_9way_masked_acc"],
                "stl10_frog_leakage_rate": challenge2["stl10_frog_leakage_rate"],
                "domain_drop_9way": challenge2["domain_drop_9way"],
                "total_train_time_sec": training["total_train_time_sec"],
                "peak_gpu_mem_mb": training["peak_gpu_mem_mb"],
                "bs1_latency_ms": benchmark["bs1_latency_ms"],
                "bs128_latency_ms_per_batch": benchmark["bs128_latency_ms_per_batch"],
                "bs128_throughput_imgs_per_sec": benchmark["bs128_throughput_imgs_per_sec"],
            }
        )
        rows.append(row)
    return rows


def build_seed_sensitivity_rows(
    summary_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Mean/sample-std/min/max over the augmented ResNet seeds 0/42/123."""
    by_exp_id = {row["exp_id"]: row for row in summary_rows}
    rows: list[dict[str, Any]] = []
    for metric in SEED_SENSITIVITY_METRICS:
        values = [float(by_exp_id[exp_id][metric]) for exp_id in SEED_SENSITIVITY_EXP_IDS]
        rows.append({"metric": metric, **summarize_seed_metrics(values)})
    return rows


def build_training_duration_rows(
    records: Sequence[dict[str, Any]],
    results_dir: str | Path,
) -> list[dict[str, Any]]:
    """Validation accuracy and cumulative train time at epochs 10/20/30 for all 5 rows."""
    rows: list[dict[str, Any]] = []
    for record in records:
        cfg = record["config"]
        history_path = Path(results_dir) / "histories" / f"{record['exp_id']}.csv"
        with history_path.open("r", encoding="utf-8", newline="") as fh:
            history = list(csv.DictReader(fh))
        val_acc_by_epoch: dict[int, float] = {}
        cumulative_by_epoch: dict[int, float] = {}
        cumulative = 0.0
        for hist_row in history:
            epoch = int(hist_row["epoch"])
            cumulative += float(hist_row["epoch_time_sec"])
            val_acc_by_epoch[epoch] = float(hist_row["val_acc"])
            cumulative_by_epoch[epoch] = cumulative
        for epoch in TRAINING_DURATION_EPOCHS:
            if epoch not in val_acc_by_epoch:
                raise RuntimeError(
                    f"{record['exp_id']}: history is missing epoch {epoch}; "
                    "cannot build the training-duration table."
                )
            rows.append(
                {
                    "exp_id": record["exp_id"],
                    "model": cfg["model"],
                    "seed": cfg["seed"],
                    "augmentation": cfg["augmentation"],
                    "epoch": epoch,
                    "val_acc": val_acc_by_epoch[epoch],
                    "cumulative_train_time_sec": cumulative_by_epoch[epoch],
                }
            )
    return rows


def build_hardware_comparison_rows(
    records: Sequence[dict[str, Any]],
    repo_results_dir: str | Path,
) -> list[dict[str, Any]]:
    """Pair Role A RTX 5070 Ti records with the matching Role C no-aug T4 rows.

    This is a cross-hardware reference table, never a same-device comparison; every row
    is explicitly labelled with its device and owner.
    """
    role_c_by_id = {record["exp_id"]: record for record in records}
    rows: list[dict[str, Any]] = []
    for role_a_id, role_c_id in HARDWARE_COMPARISON_PAIRS:
        role_a_path = Path(repo_results_dir) / f"{role_a_id}.json"
        if not role_a_path.is_file():
            raise FileNotFoundError(
                f"Role A reference record not found: {role_a_path}. The hardware "
                "comparison table requires the RTX 5070 Ti baseline JSONs."
            )
        for record in (load_json(role_a_path), role_c_by_id[role_c_id]):
            rows.append(
                {
                    "comparison_pair": record["config"]["model"],
                    "exp_id": record["exp_id"],
                    "owner": record["owner"],
                    "model": record["config"]["model"],
                    "seed": record["config"]["seed"],
                    "augmentation": record["config"]["augmentation"],
                    "device_name": record["environment"]["device_name"],
                    "training_device_name": record["environment"]["device_name"],
                    "inference_device_name": record["inference_benchmark"]["device_name"],
                    "total_train_time_sec": record["training"]["total_train_time_sec"],
                    "bs1_latency_ms": record["inference_benchmark"]["bs1_latency_ms"],
                    "bs128_throughput_imgs_per_sec": record["inference_benchmark"][
                        "bs128_throughput_imgs_per_sec"
                    ],
                    "clean_test_acc": record["evaluation"]["clean_test_acc"],
                    "comparison_note": HARDWARE_COMPARISON_NOTE,
                }
            )
    return rows


def assert_tesla_t4() -> torch.device:
    """Assert the current CUDA device is exactly a Tesla T4 and return it."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Role C stages require a CUDA Tesla T4 GPU, but CUDA is not available. "
            "Refusing to run on other hardware."
        )
    device_name = torch.cuda.get_device_name(0)
    if device_name != EXPECTED_DEVICE_NAME:
        raise RuntimeError(
            f"Role C stages must run on a Tesla T4, but the device is {device_name!r}. "
            "Refusing to substitute results from other hardware."
        )
    return torch.device("cuda:0")


def assert_jade_gpu() -> torch.device:
    """Assert the user-selected Jade evaluation GPU and return it."""
    if not torch.cuda.is_available():
        raise RuntimeError("Role C evaluation on Jade requires CUDA, but CUDA is unavailable.")
    device_name = torch.cuda.get_device_name(0)
    if device_name != EVAL_DEVICE_NAME:
        raise RuntimeError(
            f"Role C evaluation must use {EVAL_DEVICE_NAME!r} on Jade, got {device_name!r}."
        )
    return torch.device("cuda:0")


def build_experiment_config(spec: dict[str, Any], baseline_cfg: dict[str, Any]) -> ExperimentConfig:
    """Build the ExperimentConfig for one frozen matrix row from baseline.json options."""
    model_cfg = baseline_cfg["baselines"][spec["model"]]
    return ExperimentConfig(
        exp_id=spec["exp_id"],
        task=spec["task"],
        owner=EXPECTED_OWNER,
        model=spec["model"],
        model_options=dict(model_cfg["model_options"]),
        seed=int(spec["seed"]),
        epochs=int(baseline_cfg["epochs"]),
        batch_size=int(baseline_cfg["batch_size"]),
        optimizer=model_cfg["optimizer"],
        lr=float(model_cfg["lr"]),
        momentum=float(model_cfg["momentum"]),
        weight_decay=float(model_cfg["weight_decay"]),
        scheduler=model_cfg["scheduler"],
        augmentation=spec["augmentation"],
        eval_train_every=1,
        precision=EXPECTED_PRECISION,
        notes=(
            f"Role C {spec['task']} frozen T4 matrix row "
            f"({spec['model']}, seed {spec['seed']}, aug {spec['augmentation']})"
        ),
    )


def expected_record_config(spec: dict[str, Any], baseline_cfg: dict[str, Any]) -> dict[str, Any]:
    """Expected result-JSON `config` block, mirroring run_experiment's dropout injection."""
    cfg = build_experiment_config(spec, baseline_cfg)
    dropout_val = cfg.model_options.get(
        "dropout", 0.5 if cfg.model == "alexnet_cifar" else 0.0
    )
    return {**asdict(cfg), "dropout": dropout_val}


def build_train_pil_transform(augmentation: str) -> transforms.Compose | None:
    """Return the PIL train transform for a frozen augmentation name (None for no aug)."""
    if augmentation == AUGMENTATION_NONE:
        return None
    if augmentation == AUGMENTATION_CROP_FLIP:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
    raise ValueError(f"Unknown Role C augmentation: {augmentation!r}")


def _read_history_epochs(history_path: Path) -> list[int]:
    with history_path.open("r", encoding="utf-8", newline="") as fh:
        return [int(row["epoch"]) for row in csv.DictReader(fh)]


def validate_training_record(
    spec: dict[str, Any],
    baseline_cfg: dict[str, Any],
    split_stats: dict[str, Any],
    results_dir: str | Path,
    checkpoint_dir: str | Path,
) -> bool:
    """Validate one frozen-matrix training record for resume.

    Returns True only when the result JSON, per-epoch history, and checkpoint are all
    complete and exactly match the frozen spec, the frozen split stats, and a Tesla T4
    FP32 environment. Returns False when the record is missing, truncated, or corrupt
    while its stored config still matches the frozen spec (safe to retrain from scratch).
    Raises ValueError when an existing record's config/identity does not match the frozen
    spec; the original files are left untouched.
    """
    results_dir = Path(results_dir)
    checkpoint_dir = Path(checkpoint_dir)
    exp_id = spec["exp_id"]
    expected_epochs = int(baseline_cfg["epochs"])

    result_path = results_dir / f"{exp_id}.json"
    if not result_path.is_file():
        return False
    try:
        record = load_json(result_path)
    except Exception:
        return False  # Unreadable JSON: corrupt record with no verifiable config.
    if not isinstance(record, dict):
        return False

    if record.get("exp_id") != exp_id:
        raise ValueError(
            f"{exp_id}: result JSON exp_id mismatch ({record.get('exp_id')!r}); "
            "preserving the existing file. Use a fresh --work-lab1 directory."
        )
    if record.get("config") != expected_record_config(spec, baseline_cfg):
        raise ValueError(
            f"{exp_id}: stored config does not match the frozen Role C spec; "
            "preserving the existing file. Use a fresh --work-lab1 directory."
        )
    if record.get("owner") != EXPECTED_OWNER or record.get("task") != spec["task"]:
        raise ValueError(
            f"{exp_id}: stored owner/task does not match the frozen Role C spec; "
            "preserving the existing file."
        )
    environment = record.get("environment") or {}
    if (
        environment.get("device_name") != EXPECTED_DEVICE_NAME
        or environment.get("precision") != EXPECTED_PRECISION
    ):
        raise ValueError(
            f"{exp_id}: stored environment is not {EXPECTED_DEVICE_NAME}/{EXPECTED_PRECISION} "
            f"({environment.get('device_name')!r}/{environment.get('precision')!r}); "
            "T4 results are never substituted from other hardware."
        )
    training = record.get("training") or {}
    if training.get("num_trained_samples") != EXPECTED_TRAINED_SAMPLES:
        raise ValueError(
            f"{exp_id}: num_trained_samples is {training.get('num_trained_samples')!r}, "
            f"expected {EXPECTED_TRAINED_SAMPLES}; preserving the existing file."
        )

    # History must contain exactly the continuous epochs 1..30.
    history_path = results_dir / "histories" / f"{exp_id}.csv"
    if not history_path.is_file():
        return False
    try:
        epochs = _read_history_epochs(history_path)
    except Exception:
        return False
    if epochs != list(range(1, expected_epochs + 1)):
        return False

    # Checkpoint must actually load and match the record, spec, and frozen split.
    ckpt_path = checkpoint_dir / exp_id / "best.pt"
    if not ckpt_path.is_file():
        return False
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return False  # Corrupt checkpoint bytes: retrain.
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        return False

    cfg = build_experiment_config(spec, baseline_cfg)
    if payload.get("exp_id") != exp_id or payload.get("config") != asdict(cfg):
        raise ValueError(
            f"{exp_id}: checkpoint exp_id/config does not match the frozen Role C spec; "
            "preserving the existing file. Use a fresh --work-lab1 directory."
        )
    if payload.get("stats") != split_stats:
        raise ValueError(
            f"{exp_id}: checkpoint normalization stats do not match the frozen split; "
            "preserving the existing file. Use a fresh --work-lab1 directory."
        )
    for key in ("best_epoch", "best_val_acc", "best_val_ce"):
        if payload.get(key) != training.get(key):
            return False  # Internally inconsistent record: retrain.

    try:
        model, _ = build_model(cfg.model, cfg.model_options, device="cpu")
        model.load_state_dict(payload["model_state_dict"])
    except Exception:
        return False  # State dict does not fit the model: corrupt, retrain.
    return True


def find_invalid_records(
    baseline_cfg: dict[str, Any],
    split_stats: dict[str, Any],
    results_dir: str | Path,
    checkpoint_dir: str | Path,
) -> dict[str, str]:
    """Return {exp_id: reason} for every frozen row that is not a fully valid record."""
    problems: dict[str, str] = {}
    for spec in EXPERIMENT_SPECS:
        try:
            valid = validate_training_record(
                spec, baseline_cfg, split_stats, results_dir, checkpoint_dir
            )
        except ValueError as exc:
            problems[spec["exp_id"]] = str(exc)
            continue
        if not valid:
            problems[spec["exp_id"]] = "missing, truncated, or corrupt training record"
    return problems


def run_training_stage(
    repo_root: str | Path,
    work_lab1: str | Path,
    device: torch.device | str,
    show_progress: bool = False,
) -> dict[str, list[str]]:
    """Train (or resume) the five frozen matrix rows without touching test/STL data."""
    repo_root = Path(repo_root)
    work_lab1 = Path(work_lab1)
    lab1_dir = repo_root / "lab1"
    baseline_cfg = load_json(lab1_dir / "configs" / "baseline.json")

    data_dir = work_lab1 / "data"
    results_dir = work_lab1 / RESULTS_DIRNAME
    checkpoint_dir = work_lab1 / "checkpoints"
    manifest_path = results_dir / "checkpoints_manifest.json"

    train_ds, test_ds = load_cifar10_datasets(data_dir)
    split_info = get_or_create_split(train_ds, lab1_dir / "configs" / "split_indices.json")
    split_stats = split_info["stats"]
    num_workers = 0 if platform.system() == "Windows" else 2

    summary: dict[str, list[str]] = {"trained": [], "skipped": []}
    for spec in EXPERIMENT_SPECS:
        exp_id = spec["exp_id"]
        if validate_training_record(
            spec, baseline_cfg, split_stats, results_dir, checkpoint_dir
        ):
            print(f">>> Skipping {exp_id}: existing record passed full validation.")
            summary["skipped"].append(exp_id)
            continue

        cfg = build_experiment_config(spec, baseline_cfg)
        # Independent loaders per row, seeded with this row's seed.
        loaders = make_loaders(
            train_dataset=train_ds,
            test_dataset=test_ds,
            train_ids=split_info["train_ids"],
            val_ids=split_info["val_ids"],
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            train_pil_transform=build_train_pil_transform(cfg.augmentation),
            num_workers=num_workers,
        )
        print(
            f"\n>>> Starting {exp_id} ({cfg.epochs} epochs, lr={cfg.lr}, "
            f"aug={cfg.augmentation}, seed={cfg.seed}, device={device}) ..."
        )
        _, record, _ = run_experiment(
            config=cfg,
            loaders=loaders,
            stats=split_stats,
            device=device,
            results_dir=results_dir,
            checkpoint_dir=checkpoint_dir,
            manifest_path=manifest_path,
            drive_rel_prefix="checkpoints",
            evaluate_test=False,
            run_benchmark=False,
            repo_root=repo_root,
            show_progress=show_progress,
        )
        summary["trained"].append(exp_id)
        print(
            f"<<< Finished {exp_id}: best_epoch={record['training']['best_epoch']}, "
            f"best_val_acc={record['training']['best_val_acc']:.4f}, "
            f"train_sec={record['training']['total_train_time_sec']:.1f}s, "
            f"peak_mem={record['training']['peak_gpu_mem_mb']:.1f}MB"
        )

    problems = find_invalid_records(
        baseline_cfg, split_stats, results_dir, checkpoint_dir
    )
    if problems:
        raise RuntimeError(
            f"Role C training stage ended with invalid records: {problems}"
        )
    print(
        f"\nRole C training stage complete: {len(summary['trained'])} trained, "
        f"{len(summary['skipped'])} resumed."
    )
    return summary


def run_smoke_stage(
    repo_root: str | Path,
    work_lab1: str | Path,
    device: torch.device | str,
) -> dict[str, Any]:
    """Run the isolated 1-epoch smoke check (512 train / 128 val IDs, benchmark on).

    Never iterates or evaluates the CIFAR-10 test split and never loads STL-10.
    """
    repo_root = Path(repo_root)
    work_lab1 = Path(work_lab1)
    lab1_dir = repo_root / "lab1"
    baseline_cfg = load_json(lab1_dir / "configs" / "baseline.json")

    data_dir = work_lab1 / "data"
    results_dir = work_lab1 / "results" / "smoke_role_c"
    checkpoint_dir = work_lab1 / "checkpoints" / "smoke_role_c"
    manifest_path = results_dir / "checkpoints_manifest.json"

    train_ds, test_ds = load_cifar10_datasets(data_dir)
    split_info = get_or_create_split(train_ds, lab1_dir / "configs" / "split_indices.json")

    model_cfg = baseline_cfg["baselines"]["resnet18_cifar"]
    cfg = ExperimentConfig(
        exp_id=SMOKE_EXP_ID,
        task="smoke",
        owner=EXPECTED_OWNER,
        model="resnet18_cifar",
        model_options=dict(model_cfg["model_options"]),
        seed=int(baseline_cfg["default_seed"]),
        epochs=SMOKE_EPOCHS,
        batch_size=int(baseline_cfg["batch_size"]),
        optimizer=model_cfg["optimizer"],
        lr=float(model_cfg["lr"]),
        momentum=float(model_cfg["momentum"]),
        weight_decay=float(model_cfg["weight_decay"]),
        scheduler=model_cfg["scheduler"],
        augmentation=AUGMENTATION_CROP_FLIP,
        eval_train_every=1,
        precision=EXPECTED_PRECISION,
        notes=(
            "Role C smoke: 1 epoch on 512 train / 128 val IDs with crop+flip; "
            "no CIFAR test iteration, no STL-10."
        ),
    )
    num_workers = 0 if platform.system() == "Windows" else 2
    loaders = make_loaders(
        train_dataset=train_ds,
        test_dataset=test_ds,
        train_ids=split_info["train_ids"][:SMOKE_TRAIN_SAMPLES],
        val_ids=split_info["val_ids"][:SMOKE_VAL_SAMPLES],
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        train_pil_transform=build_train_pil_transform(cfg.augmentation),
        num_workers=num_workers,
    )
    print(f"\n>>> Starting smoke {SMOKE_EXP_ID} (1 epoch, device={device}) ...")
    _, record, _ = run_experiment(
        config=cfg,
        loaders=loaders,
        stats=split_info["stats"],
        device=device,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        manifest_path=manifest_path,
        drive_rel_prefix="checkpoints/smoke_role_c",
        evaluate_test=False,
        run_benchmark=True,
        repo_root=repo_root,
        show_progress=False,
    )
    trained = record["training"]["num_trained_samples"]
    if trained != SMOKE_TRAIN_SAMPLES:
        raise RuntimeError(
            f"Smoke run trained on {trained} samples, expected {SMOKE_TRAIN_SAMPLES}."
        )
    print(
        f"<<< Smoke complete: {trained} trained samples, "
        f"benchmark bs1={record['inference_benchmark'].get('bs1_latency_ms')}ms. "
        f"Results isolated under {results_dir}."
    )
    return record


def run_evaluation(
    repo_root: str | Path,
    work_lab1: str | Path,
    device: torch.device | str | None = None,
) -> list[dict[str, Any]]:
    """Run the unified Role C evaluation (Task 4 / Challenge 2 / Task 6).

    Hard gate: all five frozen training records must pass full validation before any
    CIFAR-10 test or STL-10 data is loaded. When ``device`` is None, the Tesla T4
    assertion runs only after this gate. Returns the five updated result records in
    frozen-matrix order.
    """
    repo_root = Path(repo_root)
    work_lab1 = Path(work_lab1)
    lab1_dir = repo_root / "lab1"
    baseline_cfg = load_json(lab1_dir / "configs" / "baseline.json")
    split_payload = load_json(lab1_dir / "configs" / "split_indices.json")
    split_stats = split_payload["stats"]

    results_dir = work_lab1 / RESULTS_DIRNAME
    checkpoint_dir = work_lab1 / "checkpoints"

    problems = find_invalid_records(
        baseline_cfg,
        split_stats,
        results_dir,
        checkpoint_dir,
    )
    if problems:
        raise RuntimeError(
            "Role C evaluation requires all 5 frozen training records to pass full "
            "validation before any CIFAR-10 test or STL-10 access; refusing to run. "
            f"Problems: {problems}"
        )

    # --- Gate passed: all five frozen records are complete and exactly matching. ---
    if device is None:
        device = assert_jade_gpu()
    device = torch.device(device)

    # Freeze the corruption protocol byte-exactly into the work dir; a conflicting
    # existing copy is never overwritten.
    repo_corruption_path = lab1_dir / "configs" / "corruptions_t4.json"
    config_bytes = repo_corruption_path.read_bytes()
    work_configs_dir = work_lab1 / "configs"
    work_corruption_path = work_configs_dir / "corruptions_t4.json"
    if work_corruption_path.exists():
        if work_corruption_path.read_bytes() != config_bytes:
            raise RuntimeError(
                f"Conflicting corruption config at {work_corruption_path}: existing copy "
                "differs from the frozen repo config; refusing to overwrite it."
            )
    else:
        work_configs_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = work_corruption_path.with_suffix(work_corruption_path.suffix + ".tmp")
        tmp_path.write_bytes(config_bytes)
        tmp_path.replace(work_corruption_path)
    corruption_sha256 = hashlib.sha256(config_bytes).hexdigest()
    corruption_config = load_corruption_config(work_corruption_path)

    # Data access begins only below this line.
    data_dir = work_lab1 / "data"
    _, test_ds = load_cifar10_datasets(data_dir)
    num_workers = 0 if platform.system() == "Windows" else 2
    batch_size = int(baseline_cfg["batch_size"])
    corruption_loaders = make_corruption_loaders(
        test_ds,
        corruption_config,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    # The inherited Drive data/ directory rejects new files; use an independent cache.
    stl_data_dir = work_lab1 / "data_role_c"
    stl10_loader = make_stl10_test_loader(
        stl_data_dir, batch_size=batch_size, num_workers=num_workers
    )

    condition_order: tuple[str, ...] = ("clean", *EXPECTED_CORRUPTION_NAMES)
    predictions_dir = results_dir / "predictions"
    records: list[dict[str, Any]] = []
    reference_cifar_ids: np.ndarray | None = None
    reference_stl_ids: np.ndarray | None = None

    for spec in EXPERIMENT_SPECS:
        exp_id = spec["exp_id"]
        result_path = results_dir / f"{exp_id}.json"
        record = load_json(result_path)

        # Rebuild the model from the checkpoint's own config and verify frozen stats.
        ckpt_path = checkpoint_dir / exp_id / "best.pt"
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_config = checkpoint["config"]
        if checkpoint.get("stats") != split_stats:
            raise RuntimeError(
                f"{exp_id}: checkpoint normalization stats do not match the frozen "
                "split stats; refusing to evaluate."
            )
        model, input_size = build_model(
            ckpt_config["model"], ckpt_config["model_options"], device=device
        )
        load_checkpoint_weights(model, ckpt_path, device)
        print(f"\n>>> Evaluating {exp_id} ({ckpt_config['model']}, device={device}) ...")

        # Task 4: clean + 3 fixed corruptions on identical IDs, synchronized timing.
        eval_results: dict[str, dict[str, Any]] = {}
        for condition in condition_order:
            synchronize(device)
            started = time.perf_counter()
            result = evaluate_model(
                model,
                corruption_loaders[condition],
                split_stats,
                device,
                input_size=input_size,
            )
            synchronize(device)
            result["evaluation_time_sec"] = float(time.perf_counter() - started)
            eval_results[condition] = result

        clean = eval_results["clean"]
        for condition in condition_order[1:]:
            if not np.array_equal(eval_results[condition]["ids"], clean["ids"]):
                raise RuntimeError(
                    f"{exp_id}: corruption condition {condition!r} did not preserve "
                    "the clean image ID order."
                )
        if reference_cifar_ids is None:
            reference_cifar_ids = clean["ids"]
        elif not np.array_equal(clean["ids"], reference_cifar_ids):
            raise RuntimeError(f"{exp_id}: CIFAR-10 test IDs differ from other models.")

        corrupted_metrics: dict[str, Any] = {
            "clean": {
                "accuracy": clean["accuracy"],
                "ce": clean["ce"],
                "macro_f1": clean["macro_f1"],
                "num_samples": clean["num_samples"],
                "per_class_accuracy": clean["per_class_accuracy"],
                "evaluation_time_sec": clean["evaluation_time_sec"],
            }
        }
        for condition in condition_order[1:]:
            result = eval_results[condition]
            corrupted_metrics[condition] = {
                "accuracy": result["accuracy"],
                "ce": result["ce"],
                "macro_f1": result["macro_f1"],
                "num_samples": result["num_samples"],
                "per_class_accuracy": result["per_class_accuracy"],
                "evaluation_time_sec": result["evaluation_time_sec"],
                "accuracy_drop_vs_clean": float(clean["accuracy"] - result["accuracy"]),
                **compare_corrupted_to_clean(
                    clean["targets"], clean["predictions"], result["predictions"]
                ),
            }

        # Challenge 2: CIFAR-10 shared9 + one STL-10 pass, then the cross-domain summary.
        cifar9 = evaluate_cifar10_shared9(clean)
        stl10 = evaluate_stl10_cross_domain(
            model, stl10_loader, split_stats, device, input_size=input_size
        )
        if reference_stl_ids is None:
            reference_stl_ids = stl10["ids"]
        elif not np.array_equal(stl10["ids"], reference_stl_ids):
            raise RuntimeError(f"{exp_id}: STL-10 IDs differ from other models.")
        challenge2 = build_cross_domain_summary(cifar9, stl10)

        # Task 6: synchronized FP32 inference benchmark.
        benchmark = benchmark_inference(
            model,
            split_stats,
            device,
            input_size=input_size,
            warmup_runs=10,
            measured_runs=50,
            precision=EXPECTED_PRECISION,
        )

        # Per-image predictions go to NPZ only, never into the JSON record.
        arrays: dict[str, np.ndarray] = {
            "cifar_clean_ids": clean["ids"],
            "cifar_clean_targets": clean["targets"],
            "cifar_clean_predictions": clean["predictions"],
        }
        for condition in condition_order[1:]:
            result = eval_results[condition]
            arrays[f"cifar_{condition}_ids"] = result["ids"]
            arrays[f"cifar_{condition}_targets"] = result["targets"]
            arrays[f"cifar_{condition}_predictions"] = result["predictions"]
        arrays["stl10_ids"] = stl10["ids"]
        arrays["stl10_targets"] = stl10["targets"]
        arrays["stl10_predictions_10way"] = stl10["predictions_10way"]
        arrays["stl10_predictions_9way"] = stl10["predictions_9way"]
        save_numpy_archive(arrays, predictions_dir / f"{exp_id}.npz")

        evaluation = record["evaluation"]
        evaluation["clean_test_acc"] = clean["accuracy"]
        evaluation["clean_test_ce"] = clean["ce"]
        evaluation["macro_f1"] = clean["macro_f1"]
        evaluation["corrupted_metrics"] = corrupted_metrics
        evaluation["challenge2_stl10"] = challenge2
        evaluation["prediction_file"] = f"lab1/results/predictions/{exp_id}.npz"
        evaluation["corruption_config_sha256"] = corruption_sha256
        record["inference_benchmark"] = benchmark
        save_json(record, result_path)
        records.append(record)
        print(
            f"<<< {exp_id}: clean={clean['accuracy']:.4f}, "
            f"mean_corrupted={_mean_corruption_accuracy(corrupted_metrics):.4f}, "
            f"stl9={challenge2['stl10_9way_masked_acc']:.4f}, "
            f"domain_drop={challenge2['domain_drop_9way']:.4f}, "
            f"bs1={benchmark['bs1_latency_ms']:.3f}ms"
        )

    # Role C CSV tables (the team summary.csv is never touched).
    summary_rows = build_role_c_summary_rows(records)
    save_history_csv(summary_rows, results_dir / "role_c_summary.csv")
    save_history_csv(
        build_seed_sensitivity_rows(summary_rows),
        results_dir / "task6_seed_sensitivity.csv",
    )
    save_history_csv(
        build_training_duration_rows(records, results_dir),
        results_dir / "task6_training_duration.csv",
    )
    save_history_csv(
        build_hardware_comparison_rows(records, lab1_dir / "results"),
        results_dir / "task6_hardware_comparison.csv",
    )
    print(f"\nRole C evaluation complete: 5 records updated, CSV tables in {results_dir}.")

    # Role C figures from saved JSON/CSV/NPZ artifacts only; pixels are rebuilt from
    # the already-loaded original datasets and the frozen corruption config, so no
    # additional CIFAR-10 test or STL-10 batch is ever evaluated for plots.
    figure_paths = generate_role_c_plots(
        results_dir,
        test_ds,
        stl10_loader.dataset.dataset,
        corruption_config,
    )
    figures_dir = figure_paths["t4_learning_curves.png"].parent
    print(f"Role C figures written to {figures_dir}:")
    for figure_name, figure_path in figure_paths.items():
        print(f"  {figure_name} -> {figure_path}")
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Role C frozen T4 experiment matrix runner (Task 4 / Challenge 2 / Task 6)."
    )
    parser.add_argument("--repo-root", required=True, help="Repository root (read-only configs).")
    parser.add_argument(
        "--work-lab1",
        required=True,
        help="Writable lab1 work dir with data/, checkpoints/, results_role_c/ (e.g. Drive mount).",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["smoke", "train", "evaluate", "all"],
        help="smoke: 1-epoch sanity; train: 5-row matrix; evaluate: unified eval; all: train+evaluate.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    work_lab1 = Path(args.work_lab1).resolve()

    device = assert_jade_gpu() if args.stage == "evaluate" else assert_tesla_t4()
    print(f"Role C stage={args.stage} | device={torch.cuda.get_device_name(0)} | work={work_lab1}")

    if args.stage == "smoke":
        run_smoke_stage(repo_root, work_lab1, device)
        return
    if args.stage in ("train", "all"):
        run_training_stage(repo_root, work_lab1, device)
    if args.stage in ("evaluate", "all"):
        run_evaluation(repo_root, work_lab1, device)


if __name__ == "__main__":
    main()
