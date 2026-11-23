"""Focused unit tests for Lab 1 core modules (no dataset download required)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import (  # noqa: E402
    CIFAR10_FROG_CLASS_ID,
    CIFAR10_SHARED_9_IDS,
    RawImages,
    STL10_TO_CIFAR10,
    TensorBatchLoader,
    prepare_inputs,
)
from io_utils import aggregate_results_to_csv, load_json, register_checkpoint, save_json  # noqa: E402
from metrics import evaluate_cifar10_shared9  # noqa: E402
from models import build_model, count_parameters, trace_shapes  # noqa: E402
from seed import set_seed  # noqa: E402


class _DummyPILDataset:
    def __init__(self, n: int = 8, size: int = 32) -> None:
        self.images = [
            Image.fromarray(np.full((size, size, 3), i * 20, dtype=np.uint8)) for i in range(n)
        ]
        self.targets = [i % 10 for i in range(n)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.targets[idx]


class Lab1CoreTests(unittest.TestCase):
    def test_alexnet_and_resnet18_shapes_and_params(self) -> None:
        alex, size_a = build_model("alexnet_cifar")
        self.assertEqual(size_a, 32)
        shapes_a = trace_shapes(alex, input_size=32, device="cpu")
        self.assertEqual(shapes_a[0]["shape"], [1, 64, 16, 16])
        self.assertEqual(shapes_a[1]["shape"], [1, 192, 8, 8])
        self.assertEqual(shapes_a[2]["shape"], [1, 256, 4, 4])
        self.assertEqual(shapes_a[3]["shape"], [1, 10])
        self.assertGreater(count_parameters(alex)["total_params"], 6_000_000)

        res, size_r = build_model("resnet18_cifar")
        self.assertEqual(size_r, 32)
        shapes_r = trace_shapes(res, input_size=32, device="cpu")
        self.assertEqual(shapes_r[0]["shape"], [1, 64, 32, 32])
        self.assertEqual(shapes_r[1]["shape"], [1, 64, 32, 32])
        self.assertEqual(shapes_r[2]["shape"], [1, 128, 16, 16])
        self.assertEqual(shapes_r[3]["shape"], [1, 256, 8, 8])
        self.assertEqual(shapes_r[4]["shape"], [1, 512, 4, 4])
        self.assertEqual(shapes_r[5]["shape"], [1, 10])
        self.assertEqual(count_parameters(res)["total_params"], 11_173_962)

    def test_resnet_task3_width_depth_dropout_options(self) -> None:
        narrow, _ = build_model(
            "resnet18_cifar",
            options={"layers": (1, 1, 1, 1), "base_width": 32, "dropout": 0.3},
        )
        full, _ = build_model("resnet18_cifar")
        self.assertLess(
            count_parameters(narrow)["total_params"],
            count_parameters(full)["total_params"],
        )
        out = narrow(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(out.shape), (2, 10))

    def test_prepare_inputs_differentiable_and_resizes_stl10(self) -> None:
        stats = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}
        raw = torch.rand(2, 3, 96, 96, requires_grad=True)
        normed = prepare_inputs(raw, stats, size=32)
        self.assertEqual(tuple(normed.shape), (2, 3, 32, 32))
        normed.sum().backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_raw_images_preserves_ids_and_range(self) -> None:
        ds = RawImages(_DummyPILDataset(n=6), ids=[1, 4, 5])
        tensor, target, image_id = ds[1]
        self.assertEqual(tuple(tensor.shape), (3, 32, 32))
        self.assertEqual(target, 4)
        self.assertEqual(image_id, 4)
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_tensor_batch_loader_deterministic_and_exact(self) -> None:
        imgs = torch.randint(0, 256, (10, 3, 32, 32), dtype=torch.uint8)
        targets = torch.arange(10, dtype=torch.long)
        ids = torch.arange(100, 110, dtype=torch.long)
        loader_a = TensorBatchLoader(imgs, targets, ids, batch_size=4, shuffle=True, seed=42)
        loader_b = TensorBatchLoader(imgs, targets, ids, batch_size=4, shuffle=True, seed=42)
        batches_a = list(loader_a)
        batches_b = list(loader_b)
        self.assertEqual(len(batches_a), 3)
        for (ra, ta, ia), (rb, tb, ib) in zip(batches_a, batches_b):
            self.assertTrue(torch.equal(ra, rb))
            self.assertTrue(torch.equal(ta, tb))
            self.assertTrue(torch.equal(ia, ib))
            self.assertEqual(ra.dtype, torch.float32)
            self.assertGreaterEqual(float(ra.min()), 0.0)
            self.assertLessEqual(float(ra.max()), 1.0)

    def test_stl10_label_mapping_and_shared9_metrics(self) -> None:
        self.assertEqual(len(STL10_TO_CIFAR10), 9)
        self.assertNotIn(7, STL10_TO_CIFAR10)
        self.assertEqual(set(STL10_TO_CIFAR10.values()), set(CIFAR10_SHARED_9_IDS))
        self.assertNotIn(CIFAR10_FROG_CLASS_ID, STL10_TO_CIFAR10.values())

        # Construct 2 samples: one true class 0 predicted as frog (6) with 2nd best = 0;
        # one true class 6 (frog) which must be ignored by shared9 filter.
        probs = np.zeros((2, 10), dtype=np.float32)
        probs[0, CIFAR10_FROG_CLASS_ID] = 0.7
        probs[0, 0] = 0.3
        probs[1, CIFAR10_FROG_CLASS_ID] = 0.9
        res = evaluate_cifar10_shared9(
            {"targets": np.array([0, CIFAR10_FROG_CLASS_ID]), "probabilities": probs}
        )
        self.assertEqual(res["num_samples"], 1)
        self.assertAlmostEqual(res["cifar10_shared9_10way_acc"], 0.0)
        self.assertAlmostEqual(res["cifar10_shared9_9way_masked_acc"], 1.0)
        self.assertAlmostEqual(res["cifar10_shared9_frog_leakage_rate"], 1.0)

    def test_seed_and_io_manifest_roundtrip(self) -> None:
        set_seed(42)
        a = torch.randn(4)
        set_seed(42)
        b = torch.randn(4)
        self.assertTrue(torch.equal(a, b))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest = tmp_path / "checkpoints_manifest.json"
            register_checkpoint(manifest, {"exp_id": "e1", "best_val_acc": 0.5})
            register_checkpoint(manifest, {"exp_id": "e1", "best_val_acc": 0.7})
            loaded = load_json(manifest)
            self.assertEqual(len(loaded["checkpoints"]), 1)
            self.assertAlmostEqual(loaded["checkpoints"][0]["best_val_acc"], 0.7)

            save_json(
                {
                    "exp_id": "e1",
                    "task": "task2",
                    "owner": "role_a",
                    "config": {"model": "resnet18_cifar", "seed": 42},
                    "environment": {"device_name": "CPU"},
                    "model_stats": {"total_params": 100},
                    "training": {"best_val_acc": 0.7},
                    "evaluation": {"clean_test_acc": 0.68},
                    "inference_benchmark": {"bs1_latency_ms": 1.2},
                },
                tmp_path / "e1.json",
            )
            rows = aggregate_results_to_csv(tmp_path, tmp_path / "summary.csv")
            self.assertEqual(len(rows), 1)
            self.assertTrue((tmp_path / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
