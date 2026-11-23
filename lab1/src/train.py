"""Unified training loop and experiment runner aligned with experiment_contract.md v1.0."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import prepare_inputs
from io_utils import (
    get_environment_info,
    get_git_commit,
    register_checkpoint,
    save_history_csv,
    save_json,
    utc_now_iso,
)
from metrics import benchmark_inference, evaluate_model, synchronize
from models import build_model, count_parameters
from seed import set_seed


@dataclass
class ExperimentConfig:
    """Configuration for a reproducible Lab 1 training/evaluation run."""

    exp_id: str
    task: str = "task2"
    owner: str = "role_a"
    model: str = "resnet18_cifar"
    model_options: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    epochs: int = 30
    batch_size: int = 128
    optimizer: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    scheduler: str = "cosine"
    augmentation: str = "none"
    eval_train_every: int = 1
    precision: str = "fp32"
    notes: str = ""


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: ExperimentConfig,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    """Construct optimizer and optional LR scheduler from ExperimentConfig."""
    opt_name = config.optimizer.lower()
    if opt_name == "sgd":
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")

    sched_name = config.scheduler.lower()
    if sched_name == "cosine":
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
        )
    elif sched_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler!r}")

    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    stats: dict[str, Sequence[float]],
    device: torch.device | str,
    input_size: int = 32,
    desc: str | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run one training epoch and return average cross-entropy, accuracy, and sample count."""
    dev = torch.device(device)
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    iterator = tqdm(loader, desc=desc, leave=False) if show_progress else loader
    for raw, targets, _ids in iterator:
        raw = raw.to(dev, non_blocking=True)
        targets = targets.to(dev, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(prepare_inputs(raw, stats, input_size))
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        batch_n = targets.size(0)
        loss_sum += loss.item() * batch_n
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += batch_n

    return {
        "train_ce": float(loss_sum / max(total, 1)),
        "train_acc": float(correct / max(total, 1)),
        "num_samples": int(total),
        "steps": len(loader),
    }


def _is_better_val(val_acc: float, val_ce: float, best_acc: float, best_ce: float) -> bool:
    if val_acc > best_acc:
        return True
    if val_acc == best_acc and val_ce < best_ce:
        return True
    return False


def run_experiment(
    config: ExperimentConfig,
    loaders: dict[str, DataLoader],
    stats: dict[str, Sequence[float]],
    device: torch.device | str,
    results_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    drive_rel_prefix: str = "checkpoints",
    evaluate_test: bool = False,
    run_benchmark: bool = True,
    repo_root: str | Path | None = None,
    show_progress: bool = True,
) -> tuple[nn.Module, dict[str, Any], list[dict[str, Any]]]:
    """Execute a full reproducible training, validation, logging, and optional test/benchmark run."""
    dev = torch.device(device)
    set_seed(config.seed)

    model, input_size = build_model(config.model, config.model_options, device=dev)
    param_stats = count_parameters(model)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)

    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)

    history: list[dict[str, Any]] = []
    best_val_acc = -1.0
    best_val_ce = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, Any] | None = None
    num_trained_samples = 0

    synchronize(dev)
    train_started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            stats=stats,
            device=dev,
            input_size=input_size,
            desc=f"[{epoch}/{config.epochs}] {config.exp_id}",
            show_progress=show_progress,
        )
        num_trained_samples += train_metrics["num_samples"]

        if scheduler is not None:
            scheduler.step()

        val_metrics = evaluate_model(model, loaders["val"], stats, dev, input_size=input_size)

        run_train_eval = (
            "train_eval" in loaders
            and config.eval_train_every > 0
            and (epoch % config.eval_train_every == 0 or epoch == config.epochs)
        )
        if run_train_eval:
            te_metrics = evaluate_model(
                model, loaders["train_eval"], stats, dev, input_size=input_size
            )
            train_eval_ce: float | None = te_metrics["ce"]
            train_eval_acc: float | None = te_metrics["accuracy"]
        else:
            train_eval_ce = None
            train_eval_acc = None

        synchronize(dev)
        epoch_sec = time.perf_counter() - epoch_started

        if _is_better_val(val_metrics["accuracy"], val_metrics["ce"], best_val_acc, best_val_ce):
            best_val_acc = val_metrics["accuracy"]
            best_val_ce = val_metrics["ce"]
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_ce": train_metrics["train_ce"],
                "train_acc": train_metrics["train_acc"],
                "train_eval_ce": train_eval_ce,
                "train_eval_acc": train_eval_acc,
                "val_ce": val_metrics["ce"],
                "val_acc": val_metrics["accuracy"],
                "epoch_time_sec": float(epoch_sec),
            }
        )

    synchronize(dev)
    total_train_time_sec = float(time.perf_counter() - train_started)
    peak_gpu_mem_mb = (
        float(torch.cuda.max_memory_allocated(dev) / (1024**2))
        if dev.type == "cuda" and torch.cuda.is_available()
        else 0.0
    )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    ckpt_size_mb = 0.0
    drive_rel_path: str | None = None
    registered = False
    git_commit = get_git_commit(repo_root)

    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir) / config.exp_id / "best.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "exp_id": config.exp_id,
                "config": asdict(config),
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "best_val_ce": best_val_ce,
                "stats": stats,
                "git_commit": git_commit,
                "model_state_dict": model.state_dict(),
            },
            ckpt_path,
        )
        ckpt_size_mb = float(ckpt_path.stat().st_size / (1024**2))
        drive_rel_path = f"{drive_rel_prefix}/{config.exp_id}/best.pt"

    eval_payload: dict[str, Any] = {
        "clean_test_acc": None,
        "clean_test_ce": None,
        "macro_f1": None,
        "corrupted_metrics": {},
        "adversarial_metrics": {},
        "challenge2_stl10": {},
    }
    if evaluate_test and "test" in loaders:
        test_res = evaluate_model(model, loaders["test"], stats, dev, input_size=input_size)
        eval_payload["clean_test_acc"] = test_res["accuracy"]
        eval_payload["clean_test_ce"] = test_res["ce"]
        eval_payload["macro_f1"] = test_res["macro_f1"]

    inf_payload = (
        benchmark_inference(model, stats, dev, input_size=input_size, precision=config.precision)
        if run_benchmark
        else {}
    )

    res_dir = Path(results_dir)
    history_path = res_dir / "histories" / f"{config.exp_id}.csv"
    save_history_csv(history, history_path)

    if checkpoint_dir is not None and manifest_path is not None and drive_rel_path is not None:
        register_checkpoint(
            manifest_path,
            {
                "exp_id": config.exp_id,
                "model": config.model,
                "owner": config.owner,
                "git_commit": git_commit,
                "drive_rel_path": drive_rel_path,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "clean_test_acc": eval_payload["clean_test_acc"],
                "notes": config.notes,
            },
        )
        registered = True

    final_row = history[-1] if history else {}
    config_dict = asdict(config)
    dropout_val = config.model_options.get(
        "dropout", 0.5 if config.model == "alexnet_cifar" else 0.0
    )

    result_record: dict[str, Any] = {
        "exp_id": config.exp_id,
        "task": config.task,
        "owner": config.owner,
        "git_commit": git_commit,
        "timestamp_utc": utc_now_iso(),
        "environment": get_environment_info(dev, precision=config.precision),
        "config": {**config_dict, "dropout": dropout_val},
        "model_stats": {
            **param_stats,
            "checkpoint_size_mb": round(ckpt_size_mb, 3),
        },
        "training": {
            "num_trained_samples": num_trained_samples,
            "total_train_time_sec": round(total_train_time_sec, 3),
            "avg_epoch_time_sec": round(total_train_time_sec / max(config.epochs, 1), 3),
            "peak_gpu_mem_mb": round(peak_gpu_mem_mb, 2),
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "best_val_ce": best_val_ce,
            "final_train_eval_acc": final_row.get("train_eval_acc"),
            "final_train_eval_ce": final_row.get("train_eval_ce"),
            "history_file": f"lab1/results/histories/{config.exp_id}.csv",
        },
        "evaluation": eval_payload,
        "inference_benchmark": inf_payload,
        "checkpoint_ref": {
            "drive_rel_path": drive_rel_path,
            "is_registered_in_manifest": registered,
        },
    }

    save_json(result_record, res_dir / f"{config.exp_id}.json")
    return model, result_record, history
