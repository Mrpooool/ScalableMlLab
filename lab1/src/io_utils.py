"""Atomic JSON/CSV I/O, environment capture, and checkpoint manifest management."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_git_commit(repo_root: str | Path | None = None) -> str:
    """Return the current short git commit SHA, or 'unknown' if unavailable."""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        result = subprocess.run(
            cmd,
            cwd=str(repo_root) if repo_root is not None else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def get_environment_info(
    device: torch.device | str,
    precision: str = "fp32",
    platform_name: str | None = None,
) -> dict[str, Any]:
    """Capture hardware and PyTorch environment metadata for reproducibility."""
    dev = torch.device(device)
    if platform_name is None:
        platform_name = "colab" if Path("/content").exists() else "local"
    device_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
    return {
        "platform": platform_name,
        "device_name": device_name,
        "precision": precision,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Atomically write JSON with UTF-8 encoding."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    tmp_path.replace(target)
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_history_csv(history: list[dict[str, Any]], path: str | Path) -> Path:
    """Write per-epoch training/validation history rows to CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return target
    fieldnames = list(history[0].keys())
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    tmp_path.replace(target)
    return target


def register_checkpoint(manifest_path: str | Path, entry: dict[str, Any]) -> Path:
    """Upsert a checkpoint entry (keyed by exp_id) into checkpoints_manifest.json."""
    path = Path(manifest_path)
    if path.exists():
        manifest = load_json(path)
    else:
        manifest = {"version": "1.0", "checkpoints": []}

    checkpoints = [
        item for item in manifest.get("checkpoints", []) if item.get("exp_id") != entry["exp_id"]
    ]
    checkpoints.append(entry)
    manifest["checkpoints"] = checkpoints
    return save_json(manifest, path)


def load_checkpoint_weights(
    model: torch.nn.Module,
    ckpt_path: str | Path,
    device: torch.device | str,
) -> dict[str, Any]:
    """Load saved state_dict into model and return checkpoint metadata."""
    payload = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)
    return payload


def aggregate_results_to_csv(results_dir: str | Path, out_csv: str | Path) -> list[dict[str, Any]]:
    """Aggregate all lab1/results/*.json experiment logs into a flat summary.csv."""
    res_dir = Path(results_dir)
    rows: list[dict[str, Any]] = []
    for json_path in sorted(res_dir.glob("*.json")):
        if json_path.name == "checkpoints_manifest.json":
            continue
        data = load_json(json_path)
        if "exp_id" not in data:
            continue
        cfg = data.get("config", {})
        env = data.get("environment", {})
        stats = data.get("model_stats", {})
        tr = data.get("training", {})
        ev = data.get("evaluation", {})
        inf = data.get("inference_benchmark", {})
        rows.append(
            {
                "exp_id": data.get("exp_id"),
                "task": data.get("task"),
                "owner": data.get("owner"),
                "model": cfg.get("model"),
                "seed": cfg.get("seed"),
                "epochs": cfg.get("epochs"),
                "batch_size": cfg.get("batch_size"),
                "optimizer": cfg.get("optimizer"),
                "lr": cfg.get("lr"),
                "weight_decay": cfg.get("weight_decay"),
                "dropout": cfg.get("dropout"),
                "augmentation": cfg.get("augmentation"),
                "device_name": env.get("device_name"),
                "total_params": stats.get("total_params"),
                "num_trained_samples": tr.get("num_trained_samples"),
                "total_train_time_sec": tr.get("total_train_time_sec"),
                "peak_gpu_mem_mb": tr.get("peak_gpu_mem_mb"),
                "best_epoch": tr.get("best_epoch"),
                "best_val_acc": tr.get("best_val_acc"),
                "best_val_ce": tr.get("best_val_ce"),
                "final_train_eval_acc": tr.get("final_train_eval_acc"),
                "clean_test_acc": ev.get("clean_test_acc"),
                "macro_f1": ev.get("macro_f1"),
                "bs1_latency_ms": inf.get("bs1_latency_ms"),
                "bs128_throughput_imgs_per_sec": inf.get("bs128_throughput_imgs_per_sec"),
            }
        )

    if rows:
        save_history_csv(rows, out_csv)
    return rows
