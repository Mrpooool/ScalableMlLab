"""Focused unit tests for Lab 1 core modules (no dataset download required)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from corruptions import (  # noqa: E402
    EXPECTED_CORRUPTION_NAMES,
    CorruptedImages,
    apply_corruption,
    load_corruption_config,
    make_corruption_loaders,
)
from data import (  # noqa: E402
    CIFAR10_CLASSES,
    CIFAR10_FROG_CLASS_ID,
    CIFAR10_SHARED_9_IDS,
    RawImages,
    STL10_TO_CIFAR10,
    TensorBatchLoader,
    prepare_inputs,
)
from io_utils import (  # noqa: E402
    aggregate_results_to_csv,
    load_json,
    register_checkpoint,
    save_history_csv,
    save_json,
    save_numpy_archive,
)
from metrics import (  # noqa: E402
    build_cross_domain_summary,
    compute_per_class_accuracy,
    evaluate_cifar10_shared9,
    evaluate_model,
    evaluate_stl10_cross_domain,
)
from models import build_model, count_parameters, trace_shapes  # noqa: E402
import run_role_c_experiments as role_c_runner  # noqa: E402
from run_role_c_experiments import (  # noqa: E402
    EXPERIMENT_SPECS,
    build_experiment_config,
    compare_corrupted_to_clean,
    expected_record_config,
    run_evaluation,
    summarize_seed_metrics,
    validate_training_record,
)
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


class _AsymmetricArrayDataset:
    """Numpy-backed 32x32 uint8 dataset with strongly asymmetric per-channel patterns."""

    def __init__(self, n: int = 8, size: int = 32) -> None:
        yy, xx = torch.meshgrid(
            torch.arange(size), torch.arange(size), indexing="ij"
        )
        chans = torch.stack(
            [
                (xx * 7 + 3) % 256,
                (yy * 11 + 5) % 256,
                (xx * 3 + yy * 13 + 17) % 256,
            ],
            dim=-1,
        ).to(torch.uint8)
        self.data = chans.numpy()[None, ...].repeat(n, axis=0)
        # Per-image variation so IDs are distinguishable.
        self.data = (self.data + (np.arange(n) * 29)[:, None, None, None]) % 256
        self.data = self.data.astype(np.uint8)
        self.targets = [i % 10 for i in range(n)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return Image.fromarray(self.data[idx]), self.targets[idx]


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

    def test_save_numpy_archive_roundtrip(self) -> None:
        arrays = {
            "cifar_clean_ids": np.arange(5, dtype=np.int64),
            "cifar_clean_targets": np.array([0, 1, 2, 3, 4], dtype=np.int64),
            "cifar_clean_predictions": np.array([0, 1, 2, 3, 4], dtype=np.int64),
            "stl10_predictions_9way": np.array([1, 2, 9], dtype=np.int64),
        }
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "predictions" / "exp.npz"
            out = save_numpy_archive(arrays, target)
            self.assertEqual(out, target)
            self.assertTrue(target.is_file())
            with np.load(target) as loaded:
                self.assertEqual(set(loaded.files), set(arrays))
                for key, value in arrays.items():
                    self.assertEqual(loaded[key].dtype, value.dtype)
                    self.assertTrue(np.array_equal(loaded[key], value))
            # No temporary file lingers after a successful atomic replace.
            self.assertFalse(target.with_suffix(target.suffix + ".tmp").exists())

    def test_save_numpy_archive_failed_write_preserves_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "exp.npz"
            save_numpy_archive({"ids": np.arange(3, dtype=np.int64)}, target)
            before = target.read_bytes()
            # A lambda cannot be pickled into an object array, so the write fails
            # mid-stream; the pre-existing archive must survive untouched.
            with self.assertRaises(Exception):
                save_numpy_archive({"bad": lambda: 1}, target)
            self.assertEqual(target.read_bytes(), before)
            with np.load(target) as loaded:
                self.assertTrue(np.array_equal(loaded["ids"], np.arange(3)))

    def test_compute_per_class_accuracy_covers_ten_classes_and_empty_is_none(self) -> None:
        targets = np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        predictions = np.array([0, 5, 1, 1, 7, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        per_class = compute_per_class_accuracy(targets, predictions)
        self.assertEqual(list(per_class), list(CIFAR10_CLASSES))
        self.assertAlmostEqual(per_class["airplane"], 0.5)
        self.assertAlmostEqual(per_class["automobile"], 1.0)
        self.assertAlmostEqual(per_class["bird"], 0.0)
        for name in ("cat", "deer", "dog", "frog", "horse", "ship"):
            self.assertAlmostEqual(per_class[name], 1.0)
        # Class 9 (truck) has no samples -> None.
        self.assertIsNone(per_class["truck"])

        empty = compute_per_class_accuracy(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        self.assertTrue(all(value is None for value in empty.values()))

    def test_evaluate_model_reports_per_class_accuracy(self) -> None:
        class _SequencedLogitsModel(torch.nn.Module):
            """Returns pre-scripted per-sample logits in call order (loader is unshuffled)."""

            def __init__(self, logits_per_sample) -> None:
                super().__init__()
                self._logits = [torch.tensor(l, dtype=torch.float32) for l in logits_per_sample]
                self._seen = 0

            def forward(self, x):
                start = self._seen
                self._seen += len(x)
                return torch.stack(self._logits[start : start + len(x)])

        imgs = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
        targets = torch.tensor([0, 1, 2, 2], dtype=torch.long)
        ids = torch.tensor([10, 11, 12, 13], dtype=torch.long)
        loader = TensorBatchLoader(imgs, targets, ids, batch_size=2, shuffle=False)
        stats = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
        logits_per_sample = np.zeros((4, 10), dtype=np.float32)
        logits_per_sample[np.arange(4), [0, 5, 2, 3]] = 5.0
        result = evaluate_model(
            _SequencedLogitsModel(logits_per_sample), loader, stats, device="cpu", input_size=32
        )
        self.assertEqual(list(result["per_class_accuracy"]), list(CIFAR10_CLASSES))
        self.assertAlmostEqual(result["per_class_accuracy"]["airplane"], 1.0)
        self.assertAlmostEqual(result["per_class_accuracy"]["automobile"], 0.0)
        self.assertAlmostEqual(result["per_class_accuracy"]["bird"], 0.5)
        self.assertIsNone(result["per_class_accuracy"]["cat"])
        self.assertIsNone(result["per_class_accuracy"]["truck"])
        self.assertAlmostEqual(result["accuracy"], 0.5)

    def test_shared9_per_class_acc_9way_uses_frog_masked_predictions(self) -> None:
        # Two shared samples: true classes 0 and 1; 10-way both predicted as frog,
        # masked 9-way both fall back to the true class.
        probs = np.zeros((2, 10), dtype=np.float32)
        probs[0, CIFAR10_FROG_CLASS_ID] = 0.9
        probs[0, 0] = 0.1
        probs[1, CIFAR10_FROG_CLASS_ID] = 0.8
        probs[1, 1] = 0.2
        res = evaluate_cifar10_shared9({"targets": np.array([0, 1]), "probabilities": probs})
        self.assertAlmostEqual(res["cifar10_shared9_10way_acc"], 0.0)
        self.assertAlmostEqual(res["cifar10_shared9_9way_masked_acc"], 1.0)
        self.assertEqual(res["per_class_acc_9way"], {"airplane": 1.0, "automobile": 1.0})

    def test_stl10_cross_domain_preserves_ids_and_both_prediction_views(self) -> None:
        class _SingleSTLSample:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int):
                return torch.full((3, 32, 32), 0.5), 0, 505

        class _FixedLogitsModel(torch.nn.Module):
            def __init__(self, logits) -> None:
                super().__init__()
                self.logits = torch.tensor(logits, dtype=torch.float32)

            def forward(self, x):
                return self.logits.unsqueeze(0).expand(len(x), -1)

        loader = DataLoader(_SingleSTLSample(), batch_size=1, shuffle=False)
        stats = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
        # Frog logit highest -> 10-way predicts frog (6); class 0 second -> masked predicts 0.
        logits = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]
        res = evaluate_stl10_cross_domain(_FixedLogitsModel(logits), loader, stats, device="cpu")
        self.assertEqual(res["num_samples"], 1)
        self.assertEqual(res["ids"].tolist(), [505])
        self.assertEqual(res["targets"].tolist(), [0])
        self.assertEqual(res["predictions_10way"].tolist(), [CIFAR10_FROG_CLASS_ID])
        self.assertEqual(res["predictions_9way"].tolist(), [0])
        self.assertAlmostEqual(res["stl10_10way_acc"], 0.0)
        self.assertAlmostEqual(res["stl10_frog_leakage_rate"], 1.0)
        self.assertAlmostEqual(res["stl10_9way_masked_acc"], 1.0)
        self.assertEqual(res["per_class_acc_9way"], {"airplane": 1.0})
        # No large logits retained in the returned dict.
        self.assertNotIn("logits", res)

    def test_build_cross_domain_summary_known_total_and_per_class_drops(self) -> None:
        cifar9 = {
            "num_samples": 9000,
            "cifar10_shared9_9way_masked_acc": 0.9,
            "per_class_acc_9way": {"airplane": 1.0, "bird": 0.8},
        }
        stl10 = {
            "num_samples": 7200,
            "stl10_10way_acc": 0.6,
            "stl10_frog_leakage_rate": 0.25,
            "stl10_9way_masked_acc": 0.7,
            "stl10_9way_macro_f1": 0.65,
            "per_class_acc_9way": {"airplane": 0.5, "bird": 0.9},
        }
        summary = build_cross_domain_summary(cifar9, stl10)
        self.assertEqual(summary["num_cifar10_shared9"], 9000)
        self.assertEqual(summary["num_stl10"], 7200)
        self.assertAlmostEqual(summary["cifar10_test_9way_acc"], 0.9)
        self.assertAlmostEqual(summary["stl10_9way_masked_acc"], 0.7)
        self.assertAlmostEqual(summary["domain_drop_9way"], 0.2)
        self.assertAlmostEqual(summary["per_class_domain_drop_9way"]["airplane"], 0.5)
        self.assertAlmostEqual(summary["per_class_domain_drop_9way"]["bird"], -0.1)
        self.assertAlmostEqual(summary["per_class_cifar10_9way_acc"]["bird"], 0.8)
        self.assertAlmostEqual(summary["per_class_stl10_9way_acc"]["airplane"], 0.5)
        json.dumps(summary)  # must be JSON-serializable

    def _small_corruption_config(self) -> dict:
        full = load_corruption_config(
            Path(__file__).resolve().parents[1] / "configs" / "corruptions_t4.json"
        )
        return {
            "version": full["version"],
            "seed": full["seed"],
            "dataset": full["dataset"],
            "split": full["split"],
            "image_ids": list(range(8)),
            "corruptions": full["corruptions"],
        }

    def test_corruption_config_loads_and_validates(self) -> None:
        config = load_corruption_config(
            Path(__file__).resolve().parents[1] / "configs" / "corruptions_t4.json"
        )
        self.assertEqual(config["version"], 1)
        self.assertEqual(config["seed"], 2223)
        self.assertEqual(config["dataset"], "cifar10")
        self.assertEqual(config["split"], "test")
        self.assertEqual(config["image_ids"], list(range(10000)))
        self.assertEqual(
            [spec["name"] for spec in config["corruptions"]],
            list(EXPECTED_CORRUPTION_NAMES),
        )
        self.assertEqual(
            [spec["kind"] for spec in config["corruptions"]],
            ["rotation", "translation", "gaussian_noise"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            bad_kind = self._small_corruption_config()
            bad_kind["corruptions"][0]["kind"] = "motion_blur"
            bad_path = tmp_path / "bad_kind.json"
            bad_path.write_text(json.dumps(bad_kind), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_corruption_config(bad_path)

            unsorted = self._small_corruption_config()
            unsorted["image_ids"] = list(range(9999, -1, -1))
            unsorted_path = tmp_path / "unsorted.json"
            unsorted_path.write_text(json.dumps(unsorted), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_corruption_config(unsorted_path)

            dup = self._small_corruption_config()
            dup["image_ids"] = [0] * 10000
            dup_path = tmp_path / "dup.json"
            dup_path.write_text(json.dumps(dup), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_corruption_config(dup_path)

    def test_apply_corruption_deterministic_seed_sensitive_and_clamped(self) -> None:
        raw = torch.rand(3, 32, 32)
        raw[0, 0, 0] = 0.0  # exercise lower clamp
        raw[1, 5, 7] = 1.0  # exercise upper clamp
        noise_spec = {"name": "gaussian_noise_std_0_05", "kind": "gaussian_noise", "std": 0.05}

        first = apply_corruption(raw, image_id=3, spec=noise_spec, seed=2223)
        second = apply_corruption(raw, image_id=3, spec=noise_spec, seed=2223)
        self.assertTrue(torch.equal(first, second))

        other_id = apply_corruption(raw, image_id=4, spec=noise_spec, seed=2223)
        self.assertFalse(torch.equal(first, other_id))

        other_seed = apply_corruption(raw, image_id=3, spec=noise_spec, seed=2224)
        self.assertFalse(torch.equal(first, other_seed))

        for out in (first, other_id, other_seed):
            self.assertEqual(tuple(out.shape), (3, 32, 32))
            self.assertEqual(out.dtype, torch.float32)
            self.assertGreaterEqual(float(out.min()), 0.0)
            self.assertLessEqual(float(out.max()), 1.0)
        # Exact 0/1 pixels with large noise must hit the clamps for some samples.
        raw_extreme = torch.full((3, 32, 32), 0.0)
        out_low = apply_corruption(raw_extreme, image_id=0, spec=noise_spec, seed=2223)
        self.assertGreaterEqual(float(out_low.min()), 0.0)
        raw_extreme.fill_(1.0)
        out_high = apply_corruption(raw_extreme, image_id=0, spec=noise_spec, seed=2223)
        self.assertLessEqual(float(out_high.max()), 1.0)

    def test_corruption_loaders_bit_identical_across_workers(self) -> None:
        config = self._small_corruption_config()
        dataset = _AsymmetricArrayDataset(n=8)
        loaders_w0 = make_corruption_loaders(
            dataset, config, batch_size=3, num_workers=0, pin_memory=False
        )
        loaders_w2 = make_corruption_loaders(
            dataset, config, batch_size=3, num_workers=2, pin_memory=False
        )
        self.assertEqual(
            set(loaders_w0), {"clean", *EXPECTED_CORRUPTION_NAMES}
        )
        self.assertEqual(set(loaders_w0), set(loaders_w2))

        for name in loaders_w0:
            batches_w0 = list(loaders_w0[name])
            batches_w2 = list(loaders_w2[name])
            self.assertEqual(len(batches_w0), len(batches_w2))
            seen_ids: list[int] = []
            for (raw0, tgt0, id0), (raw2, tgt2, id2) in zip(batches_w0, batches_w2):
                self.assertTrue(torch.equal(raw0, raw2))
                self.assertTrue(torch.equal(tgt0, tgt2))
                self.assertTrue(torch.equal(id0, id2))
                self.assertGreaterEqual(float(raw0.min()), 0.0)
                self.assertLessEqual(float(raw0.max()), 1.0)
                seen_ids.extend(int(i) for i in id0)
            self.assertEqual(seen_ids, list(range(8)))  # fixed ascending order, all IDs
            self.assertEqual([int(t) for t in torch.cat([b[1] for b in batches_w0])],
                             [i % 10 for i in range(8)])

    def test_all_config_corruptions_alter_asymmetric_input(self) -> None:
        config = self._small_corruption_config()
        dataset = _AsymmetricArrayDataset(n=8)
        base = RawImages(dataset, ids=config["image_ids"])
        for spec in config["corruptions"]:
            corrupted = CorruptedImages(dataset, config["image_ids"], spec, config["seed"])
            self.assertEqual(len(corrupted), len(base))
            altered = 0
            for idx in range(len(base)):
                raw, target_raw, image_id_raw = base[idx]
                out, target_out, image_id_out = corrupted[idx]
                self.assertEqual(target_out, target_raw)
                self.assertEqual(image_id_out, image_id_raw)
                self.assertEqual(tuple(out.shape), tuple(raw.shape))
                self.assertGreaterEqual(float(out.min()), 0.0)
                self.assertLessEqual(float(out.max()), 1.0)
                changed = not torch.equal(out, raw)
                if changed:
                    altered += 1
                if spec["kind"] == "gaussian_noise":
                    self.assertTrue(changed, msg=f"noise did not alter image ID {image_id_raw}")
            # Noise alters every image; geometric specs may be identity for a few IDs
            # but must change the asymmetric input for at least one sampled angle/shift.
            self.assertGreaterEqual(altered, 1, msg=f"corruption {spec['name']} never altered input")


class RoleCRunnerTests(unittest.TestCase):
    """No-GPU tests for the Role C frozen-matrix runner and resume validation."""

    def setUp(self) -> None:
        self.baseline_cfg = load_json(
            Path(__file__).resolve().parents[1] / "configs" / "baseline.json"
        )
        self.split_stats = {
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.247, 0.2435, 0.2616],
        }
        self.spec = EXPERIMENT_SPECS[0]  # t6_alexnet_cifar_no_aug_s42

    def _validate(self, work: Path) -> bool:
        return validate_training_record(
            self.spec,
            self.baseline_cfg,
            self.split_stats,
            work / "results",
            work / "checkpoints",
        )

    def _write_valid_record(self, work: Path) -> None:
        """Synthesize a complete, exactly matching training record (no GPU/training)."""
        spec = self.spec
        cfg = build_experiment_config(spec, self.baseline_cfg)
        results_dir = work / "results"
        checkpoint_dir = work / "checkpoints"

        history = [
            {
                "epoch": epoch,
                "lr": 0.1,
                "train_ce": 1.0,
                "train_acc": 0.4,
                "train_eval_ce": 1.0,
                "train_eval_acc": 0.4,
                "val_ce": 1.0,
                "val_acc": 0.4,
                "epoch_time_sec": 1.0,
            }
            for epoch in range(1, 31)
        ]
        save_history_csv(history, results_dir / "histories" / f"{spec['exp_id']}.csv")

        model, _ = build_model(cfg.model, cfg.model_options, device="cpu")
        ckpt_path = checkpoint_dir / spec["exp_id"] / "best.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "exp_id": spec["exp_id"],
                "config": asdict(cfg),
                "best_epoch": 30,
                "best_val_acc": 0.4,
                "best_val_ce": 1.0,
                "stats": self.split_stats,
                "git_commit": "unknown",
                "model_state_dict": model.state_dict(),
            },
            ckpt_path,
        )

        record = {
            "exp_id": spec["exp_id"],
            "task": spec["task"],
            "owner": "role_c",
            "config": expected_record_config(spec, self.baseline_cfg),
            "environment": {
                "platform": "colab",
                "device_name": "Tesla T4",
                "precision": "fp32",
                "torch_version": torch.__version__,
                "cuda_version": "12.4",
            },
            "training": {
                "num_trained_samples": 1_350_000,
                "total_train_time_sec": 300.0,
                "best_epoch": 30,
                "best_val_acc": 0.4,
                "best_val_ce": 1.0,
            },
            "evaluation": {"clean_test_acc": None},
        }
        save_json(record, results_dir / f"{spec['exp_id']}.json")

    def test_role_c_frozen_matrix_matches_plan(self) -> None:
        self.assertEqual(
            [spec["exp_id"] for spec in EXPERIMENT_SPECS],
            [
                "t6_alexnet_cifar_no_aug_s42",
                "t4_resnet18_cifar_no_aug_s42",
                "t4_resnet18_cifar_crop_flip_s42",
                "t6_resnet18_cifar_crop_flip_s0",
                "t6_resnet18_cifar_crop_flip_s123",
            ],
        )
        self.assertEqual([spec["task"] for spec in EXPERIMENT_SPECS],
                         ["task6", "task4", "task4", "task6", "task6"])
        self.assertEqual(
            [spec["model"] for spec in EXPERIMENT_SPECS],
            ["alexnet_cifar", "resnet18_cifar", "resnet18_cifar",
             "resnet18_cifar", "resnet18_cifar"],
        )
        self.assertEqual([spec["seed"] for spec in EXPERIMENT_SPECS], [42, 42, 42, 0, 123])
        self.assertEqual(
            [spec["augmentation"] for spec in EXPERIMENT_SPECS],
            ["none", "none", "crop32_pad4_hflip05",
             "crop32_pad4_hflip05", "crop32_pad4_hflip05"],
        )

    def test_role_c_validate_accepts_complete_matching_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            self._write_valid_record(work)
            self.assertTrue(self._validate(work))

    def test_role_c_validate_rejects_truncated_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            self._write_valid_record(work)
            history_path = work / "results" / "histories" / f"{self.spec['exp_id']}.csv"
            lines = history_path.read_text(encoding="utf-8").splitlines()
            # Keep header + only the first 25 of 30 epoch rows.
            history_path.write_text("\n".join(lines[:26]) + "\n", encoding="utf-8")
            self.assertFalse(self._validate(work))

    def test_role_c_validate_rejects_corrupted_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            self._write_valid_record(work)
            ckpt_path = work / "checkpoints" / self.spec["exp_id"] / "best.pt"
            ckpt_path.write_bytes(b"\x00\x01not-a-real-checkpoint")
            self.assertFalse(self._validate(work))

    def test_role_c_validate_raises_and_preserves_mismatched_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            self._write_valid_record(work)
            record_path = work / "results" / f"{self.spec['exp_id']}.json"
            record = load_json(record_path)
            record["config"]["seed"] = 7  # foreign config under the same exp_id
            save_json(record, record_path)
            before = record_path.read_bytes()

            with self.assertRaises(ValueError):
                self._validate(work)

            # Original (mismatched) files are preserved untouched.
            self.assertEqual(record_path.read_bytes(), before)
            self.assertTrue(
                (work / "checkpoints" / self.spec["exp_id"] / "best.pt").is_file()
            )

    def test_role_c_validate_raises_on_non_t4_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            self._write_valid_record(work)
            record_path = work / "results" / f"{self.spec['exp_id']}.json"
            record = load_json(record_path)
            record["environment"]["device_name"] = "NVIDIA GeForce RTX 5070 Ti"
            save_json(record, record_path)
            with self.assertRaises(ValueError):
                self._validate(work)

    def test_role_c_evaluate_gate_rejects_before_touching_test_or_stl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_root = tmp_path / "repo"
            configs_dir = repo_root / "lab1" / "configs"
            configs_dir.mkdir(parents=True)
            save_json(self.baseline_cfg, configs_dir / "baseline.json")
            save_json(
                {
                    "split_seed": 2223,
                    "val_ratio": 0.1,
                    "stats": self.split_stats,
                    "train_ids": [],
                    "val_ids": [],
                },
                configs_dir / "split_indices.json",
            )
            work = tmp_path / "work"  # empty: none of the 5 records exist

            with mock.patch.object(
                role_c_runner, "load_cifar10_datasets"
            ) as cifar_mock, mock.patch.object(
                role_c_runner, "make_stl10_test_loader"
            ) as stl_mock:
                with self.assertRaises(RuntimeError):
                    run_evaluation(repo_root, work)

            cifar_mock.assert_not_called()
            stl_mock.assert_not_called()

    def test_compare_corrupted_to_clean_known_values(self) -> None:
        targets = np.array([0, 1, 2, 3], dtype=np.int64)
        clean_preds = np.array([0, 1, 2, 2], dtype=np.int64)  # clean-correct: idx 0,1,2
        corrupted_preds = np.array([0, 5, 7, 2], dtype=np.int64)
        out = compare_corrupted_to_clean(targets, clean_preds, corrupted_preds)
        self.assertAlmostEqual(out["prediction_flip_rate_vs_clean"], 0.5)  # idx 1,2 flip
        self.assertEqual(out["clean_correct_to_wrong_count"], 2)  # idx 1,2
        self.assertAlmostEqual(out["clean_correct_to_wrong_rate"], 2.0 / 3.0)
        json.dumps(out)  # must be JSON-serializable

    def test_compare_corrupted_to_clean_zero_clean_correct_rate_is_none(self) -> None:
        out = compare_corrupted_to_clean(
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([2], dtype=np.int64),
        )
        self.assertEqual(out["clean_correct_to_wrong_count"], 0)
        self.assertIsNone(out["clean_correct_to_wrong_rate"])
        self.assertAlmostEqual(out["prediction_flip_rate_vs_clean"], 1.0)
        json.dumps(out)

    def test_compare_corrupted_to_clean_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            compare_corrupted_to_clean(
                np.array([0, 1]), np.array([0, 1]), np.array([0])
            )

    def test_summarize_seed_metrics_mean_sample_std_min_max(self) -> None:
        out = summarize_seed_metrics([1.0, 2.0, 3.0])
        self.assertAlmostEqual(out["mean"], 2.0)
        self.assertAlmostEqual(out["sample_std"], 1.0)  # ddof=1 sample std
        self.assertAlmostEqual(out["min"], 1.0)
        self.assertAlmostEqual(out["max"], 3.0)
        json.dumps(out)

        single = summarize_seed_metrics([0.5])
        self.assertAlmostEqual(single["mean"], 0.5)
        self.assertAlmostEqual(single["sample_std"], 0.0)
        self.assertAlmostEqual(single["min"], 0.5)
        self.assertAlmostEqual(single["max"], 0.5)

        with self.assertRaises(ValueError):
            summarize_seed_metrics([])


if __name__ == "__main__":
    unittest.main()
