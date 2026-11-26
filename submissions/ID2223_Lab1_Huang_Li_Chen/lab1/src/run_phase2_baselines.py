"""Phase 2 (Role A) runner: Task 1 CIFAR-10 EDA + Task 2 AlexNet & ResNet-18 30-epoch baselines."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "lab1" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import (  # noqa: E402
    CIFAR10_CLASSES,
    get_or_create_split,
    load_cifar10_datasets,
    make_loaders,
)
from io_utils import aggregate_results_to_csv, load_json, save_json  # noqa: E402
from models import build_model, count_parameters, trace_shapes  # noqa: E402
from plots import (  # noqa: E402
    plot_baseline_comparison,
    plot_baseline_learning_curves,
    plot_class_distribution,
    plot_sample_grid,
)
from train import ExperimentConfig, run_experiment  # noqa: E402


def run_task1_eda(train_ds, test_ds, split_info: dict, results_dir: Path) -> dict:
    """Compute Task 1 dataset statistics and save EDA figures."""
    train_ids = np.asarray(split_info["train_ids"], dtype=np.int64)
    val_ids = np.asarray(split_info["val_ids"], dtype=np.int64)
    all_train_targets = np.asarray(train_ds.targets, dtype=np.int64)
    test_targets = np.asarray(test_ds.targets, dtype=np.int64)

    train_targets = all_train_targets[train_ids]
    val_targets = all_train_targets[val_ids]

    fig_dir = results_dir / "figures"
    dist_fig = plot_class_distribution(
        train_targets, val_targets, test_targets, CIFAR10_CLASSES, fig_dir / "t1_class_distribution.png"
    )
    grid_fig = plot_sample_grid(
        train_ds.data[train_ids],
        train_targets,
        train_ids,
        CIFAR10_CLASSES,
        fig_dir / "t1_sample_grid.png",
        per_class=3,
    )

    summary = {
        "dataset": "CIFAR-10",
        "image_shape_hwc": list(train_ds.data.shape[1:]),
        "dtype": str(train_ds.data.dtype),
        "pixel_min": int(train_ds.data.min()),
        "pixel_max": int(train_ds.data.max()),
        "split_seed": split_info["split_seed"],
        "val_ratio": split_info["val_ratio"],
        "counts": {
            "official_train_total": int(len(train_ds)),
            "train_split": int(len(train_ids)),
            "val_split": int(len(val_ids)),
            "official_test": int(len(test_ds)),
        },
        "class_distribution": {
            cls: {
                "train": int(np.sum(train_targets == idx)),
                "val": int(np.sum(val_targets == idx)),
                "test": int(np.sum(test_targets == idx)),
            }
            for idx, cls in enumerate(CIFAR10_CLASSES)
        },
        "train_only_stats": split_info["stats"],
        "figures": {
            "class_distribution": str(dist_fig.relative_to(REPO_ROOT).as_posix()),
            "sample_grid": str(grid_fig.relative_to(REPO_ROOT).as_posix()),
        },
    }
    save_json(summary, results_dir / "task1_data_summary.json")
    return summary


def run_task2_baselines(
    train_ds,
    test_ds,
    split_info: dict,
    baseline_cfg: dict,
    device: torch.device,
    results_dir: Path,
    checkpoint_dir: Path,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Trace architectures and run full 30-epoch baseline training for AlexNet & ResNet-18."""
    stats = split_info["stats"]
    train_ids = split_info["train_ids"]
    val_ids = split_info["val_ids"]
    num_workers = 0 if platform.system() == "Windows" else 2

    arch_profiles = {}
    for model_name in ("alexnet_cifar", "resnet18_cifar"):
        mcfg = baseline_cfg["baselines"][model_name]
        probe, input_size = build_model(model_name, mcfg["model_options"], device=device)
        arch_profiles[model_name] = {
            "input_size": input_size,
            "parameters": count_parameters(probe),
            "stage_shapes": trace_shapes(probe, input_size=input_size, device=device),
        }
        del probe
    save_json(arch_profiles, results_dir / "task2_architecture_profiles.json")

    manifest_path = results_dir / "checkpoints_manifest.json"
    records: list[dict] = []
    histories: dict[str, list[dict]] = {}

    for model_name in ("alexnet_cifar", "resnet18_cifar"):
        mcfg = baseline_cfg["baselines"][model_name]
        loaders = make_loaders(
            train_dataset=train_ds,
            test_dataset=test_ds,
            train_ids=train_ids,
            val_ids=val_ids,
            batch_size=baseline_cfg["batch_size"],
            seed=baseline_cfg["default_seed"],
            num_workers=num_workers,
        )
        exp_id = f"t2_{model_name}_baseline_s{baseline_cfg['default_seed']}"
        cfg = ExperimentConfig(
            exp_id=exp_id,
            task="task2",
            owner="role_a",
            model=model_name,
            model_options=mcfg["model_options"],
            seed=baseline_cfg["default_seed"],
            epochs=baseline_cfg["epochs"],
            batch_size=baseline_cfg["batch_size"],
            optimizer=mcfg["optimizer"],
            lr=mcfg["lr"],
            momentum=mcfg["momentum"],
            weight_decay=mcfg["weight_decay"],
            scheduler=mcfg["scheduler"],
            augmentation="none",
            eval_train_every=1,
            notes=f"Task 2 baseline {model_name} ({baseline_cfg['epochs']} epochs, no augmentation)",
        )
        print(f"\n>>> Starting {exp_id} ({cfg.epochs} epochs, lr={cfg.lr}, device={device}) ...")
        _, record, history = run_experiment(
            config=cfg,
            loaders=loaders,
            stats=stats,
            device=device,
            results_dir=results_dir,
            checkpoint_dir=checkpoint_dir,
            manifest_path=manifest_path,
            drive_rel_prefix="checkpoints",
            evaluate_test=True,
            run_benchmark=True,
            repo_root=REPO_ROOT,
            show_progress=False,
        )
        records.append(record)
        histories[model_name] = history
        print(
            f"<<< Finished {exp_id}: "
            f"best_epoch={record['training']['best_epoch']}, "
            f"best_val_acc={record['training']['best_val_acc']:.4f}, "
            f"clean_test_acc={record['evaluation']['clean_test_acc']:.4f}, "
            f"macro_f1={record['evaluation']['macro_f1']:.4f}, "
            f"train_sec={record['training']['total_train_time_sec']:.1f}s, "
            f"peak_mem={record['training']['peak_gpu_mem_mb']:.1f}MB"
        )

    fig_dir = results_dir / "figures"
    plot_baseline_learning_curves(histories, fig_dir / "t2_learning_curves.png")
    plot_baseline_comparison(records, fig_dir / "t2_baseline_comparison.png")
    aggregate_results_to_csv(results_dir, results_dir / "summary.csv")
    return records, histories


def main() -> None:
    lab1_dir = REPO_ROOT / "lab1"
    data_dir = lab1_dir / "data"
    results_dir = lab1_dir / "results"
    checkpoint_dir = lab1_dir / "checkpoints"

    train_ds, test_ds = load_cifar10_datasets(data_dir)
    split_info = get_or_create_split(train_ds, lab1_dir / "configs" / "split_indices.json")
    baseline_cfg = load_json(lab1_dir / "configs" / "baseline.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")

    t1_summary = run_task1_eda(train_ds, test_ds, split_info, results_dir)
    print("Task 1 EDA complete:", t1_summary["counts"])

    run_task2_baselines(
        train_ds=train_ds,
        test_ds=test_ds,
        split_info=split_info,
        baseline_cfg=baseline_cfg,
        device=device,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
    )
    print("\nPhase 2 complete! Summary saved to lab1/results/summary.csv")


if __name__ == "__main__":
    main()
