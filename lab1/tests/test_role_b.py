"""Task-specific correctness tests: gradient space, budgets, ASR and isolation."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial import evaluate_attack, fgsm, random_linf, summarize_attack
from io_utils import load_json
from run_role_b_experiments import MATRIX_PATH, build_config

STATS = {"mean": [0.1, 0.2, 0.3], "std": [0.2, 0.3, 0.4]}


class PixelClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, normalized):
        # Recover raw pixels: changing preprocessing must still give raw-space FGSM.
        mean = normalized.new_tensor(STATS["mean"])[None, :, None, None]
        std = normalized.new_tensor(STATS["std"])[None, :, None, None]
        score = ((normalized * std + mean).mean((1, 2, 3)) - 0.5) * self.scale
        return torch.stack([score, -score], 1)


class RoleBTests(unittest.TestCase):
    def test_fgsm_raw_space_and_parameter_gradients(self):
        model = PixelClassifier().eval()
        model.scale.grad = torch.tensor(7.0)
        original = torch.full((1, 3, 32, 32), 0.75)
        before = copy.deepcopy(model.state_dict())
        adv = fgsm(model, original, torch.tensor([0]), STATS, 0.4)
        self.assertTrue(torch.allclose(adv, torch.full_like(adv, 0.35), atol=1e-6))
        self.assertLessEqual(float((adv - original).abs().max()), 0.4 + 1e-6)
        self.assertEqual(model.scale.grad.item(), 7.0)
        self.assertTrue(torch.equal(model.state_dict()["scale"], before["scale"]))
        self.assertFalse(adv.requires_grad)
        self.assertTrue(torch.equal(fgsm(model, original, torch.tensor([0]), STATS, 0), original))
        with self.assertRaises(ValueError):
            fgsm(model, original, torch.tensor([0]), STATS, -0.1)

    def test_asr_denominator_and_empty_clean_correct(self):
        result = summarize_attack([0, 1, 1], [1, 0, 1], [0, 1, 0])
        self.assertEqual(result["clean_correct_count"], 2)
        self.assertEqual(result["successful_count"], 2)
        self.assertEqual(result["asr"], 1.0)
        self.assertEqual(result["accuracy"], 0.0)
        self.assertIsNone(summarize_attack([1], [1], [0])["asr"])

    def test_full_attack_ids_zero_epsilon_and_mode(self):
        raw = torch.stack([torch.full((3, 32, 32), value) for value in (0.75, 0.25, 0.25)])
        loader = DataLoader(TensorDataset(raw, torch.tensor([0, 1, 0]), torch.tensor([8, 2, 9])), batch_size=2)
        model = PixelClassifier().train()
        metrics, arrays = evaluate_attack(model, loader, STATS, "cpu", 0.4)
        self.assertTrue(model.training)
        self.assertEqual(metrics["asr"], 1)
        np.testing.assert_array_equal(arrays["ids"], [8, 2, 9])
        zero, arrays = evaluate_attack(model, loader, STATS, "cpu", 0)
        self.assertEqual(zero["asr"], 0)
        np.testing.assert_array_equal(arrays["predictions"], arrays["clean_predictions"])

    def test_random_noise_is_per_id_and_does_not_use_global_rng(self):
        raw = torch.full((2, 3, 32, 32), 0.5)
        state = torch.random.get_rng_state()
        a = random_linf(raw, [20, 7], 8 / 255)
        b = random_linf(raw.flip(0), [7, 20], 8 / 255)
        self.assertTrue(torch.equal(a, b.flip(0)))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertLessEqual(float((a - raw).abs().max()), 8 / 255 + 1e-6)

    def test_matrix_has_controlled_comparisons_and_dropout_conditions(self):
        matrix = load_json(MATRIX_PATH)
        configs = [build_config(row) for row in matrix["experiments"]]
        reference = configs[0]
        for cfg in configs:
            self.assertEqual((cfg.epochs, cfg.batch_size, cfg.seed, cfg.lr), (30, 128, 42, 0.01))
            self.assertEqual(cfg.augmentation, "none")
            self.assertEqual(cfg.eval_train_every, 1)
        dropouts = {c.model_options["dropout"] for c in configs}
        self.assertEqual(dropouts, {0, 0.25, 0.5})
        for cfg in configs[1:]:
            changes = sum([cfg.model_options["width_mult"] != reference.model_options["width_mult"],
                           cfg.model_options["dropout"] != reference.model_options["dropout"],
                           cfg.weight_decay != reference.weight_decay])
            self.assertEqual(changes, 1)


if __name__ == "__main__":
    unittest.main()
