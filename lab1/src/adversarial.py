"""Raw-pixel FGSM and paired, equal-budget random-noise evaluation for Task 5."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from data import prepare_inputs
from metrics import synchronize


def fgsm(model, raw, targets, stats, epsilon):
    """Untargeted L-infinity FGSM; parameter gradients and weights are unchanged.

    Caller sets eval mode. Normalization remains in the differentiable forward;
    epsilon is measured in raw [0, 1] pixels, not normalized coordinates.
    """
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    if epsilon == 0:
        return raw.detach().clone()
    with torch.enable_grad():
        inputs = raw.detach().clone().requires_grad_(True)
        loss = nn.functional.cross_entropy(model(prepare_inputs(inputs, stats)), targets)
        gradient, = torch.autograd.grad(loss, inputs)
    return (inputs.detach() + epsilon * gradient.sign()).clamp(0, 1)


def random_linf(raw, ids, epsilon, seed=2223):
    """Uniform [-epsilon, epsilon] noise, reproducible per original image ID."""
    noise = torch.stack([
        torch.rand(raw.shape[1:], generator=torch.Generator().manual_seed(seed + int(i)))
        for i in ids
    ]).to(device=raw.device, dtype=raw.dtype)
    return (raw + epsilon * (2 * noise - 1)).clamp(0, 1)


def summarize_attack(clean_predictions, predictions, targets):
    """ASR counts only originally correct samples; accuracy counts all samples."""
    clean_correct = np.asarray(clean_predictions) == np.asarray(targets)
    wrong = np.asarray(predictions) != np.asarray(targets)
    denominator = int(clean_correct.sum())
    successful = int((clean_correct & wrong).sum())
    return {
        "num_samples": int(len(targets)),
        "clean_accuracy": float(clean_correct.mean()),
        "accuracy": float((~wrong).mean()),
        "clean_correct_count": denominator,
        "successful_count": successful,
        "asr": successful / denominator if denominator else None,
    }


def evaluate_attack(model, loader, stats, device, epsilon, method="fgsm", seed=2223):
    """Evaluate every image, preserving IDs and returning JSON metrics + arrays."""
    if method not in ("fgsm", "random_linf"):
        raise ValueError(f"Unsupported perturbation: {method}")
    was_training = model.training
    model.eval()
    batches = {key: [] for key in ("ids", "targets", "clean_predictions", "predictions", "probabilities")}
    maximum_norm = loss_sum = 0.0
    synchronize(device)
    started = time.perf_counter()
    try:
        for raw, targets, ids in loader:
            raw, targets = raw.to(device), targets.to(device)
            with torch.no_grad():
                clean_predictions = model(prepare_inputs(raw, stats)).argmax(1)
            perturbed = (fgsm(model, raw, targets, stats, epsilon) if method == "fgsm"
                         else random_linf(raw, ids, epsilon, seed))
            norm = float((perturbed - raw).abs().max().item())
            if norm > epsilon + 1e-6 or perturbed.min() < 0 or perturbed.max() > 1:
                raise AssertionError("Perturbation violates the raw-pixel constraint")
            maximum_norm = max(maximum_norm, norm)
            with torch.no_grad():
                logits = model(prepare_inputs(perturbed, stats))
                loss_sum += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
                probabilities = logits.softmax(1)
            values = (ids, targets, clean_predictions, logits.argmax(1), probabilities)
            for key, value in zip(batches, values):
                batches[key].append(value.detach().cpu().numpy())
    finally:
        model.train(was_training)
    synchronize(device)
    arrays = {key: np.concatenate(value) for key, value in batches.items()}
    metrics = summarize_attack(arrays["clean_predictions"], arrays["predictions"], arrays["targets"])
    metrics.update({
        "method": method, "targeted": False, "norm": "linf", "epsilon": epsilon,
        "max_observed_linf": maximum_norm, "ce": loss_sum / metrics["num_samples"],
        "evaluation_time_sec": time.perf_counter() - started,
        "random_seed": seed if method == "random_linf" else None,
    })
    return metrics, arrays
