"""
Contract and unit tests for stage2_ensemble_val.py: WiredEnsembleEN, WiredEnsembleAdapter,
EnsembleLike protocol compliance, and baseline evaluation utilities.
"""
import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble_types import EnsembleLike
from stage2_ensemble_val import WiredEnsembleEN, WiredEnsembleAdapter, evaluate_ensemble_disagreement


class TestWiredEnsemble(unittest.TestCase):
    def test_wired_ensemble_satisfies_protocol(self):
        ens = WiredEnsembleEN(in_dim=37, out_dim=8, M=5, base_seed=42)
        self.assertTrue(isinstance(ens, EnsembleLike))
        self.assertEqual(ens.M, 5)

    def test_wired_ensemble_forward_shapes(self):
        ens = WiredEnsembleEN(in_dim=37, out_dim=8, M=3, base_seed=42)
        x = torch.randn(4, 37)
        mu, var = ens(x)
        self.assertEqual(mu.shape, (3, 4, 8))
        self.assertEqual(var.shape, (3, 4, 8))

    def test_wired_ensemble_summarize(self):
        ens = WiredEnsembleEN(in_dim=37, out_dim=8, M=5, base_seed=123)
        x = torch.randn(4, 37)
        mu_bar, aleatoric, epistemic = ens.summarize(x)
        self.assertEqual(mu_bar.shape, (4, 8))
        self.assertEqual(aleatoric.shape, (4, 8))
        self.assertEqual(epistemic.shape, (4, 8))
        self.assertTrue((epistemic >= 0).all())

    def test_adapter_interface(self):
        ens = WiredEnsembleEN(in_dim=37, out_dim=8, M=5, base_seed=42)
        adapter = WiredEnsembleAdapter(ens)
        x = torch.randn(4, 37)
        out = adapter(x)
        self.assertEqual(out.shape, (4, 8))

    def test_disagreement_calculation(self):
        ens = WiredEnsembleEN(in_dim=37, out_dim=8, M=4, base_seed=99)
        adapter = WiredEnsembleAdapter(ens)
        dis = evaluate_ensemble_disagreement(adapter, batch_size=8)
        self.assertIsNotNone(dis)
        self.assertGreaterEqual(dis, 0.0)


if __name__ == "__main__":
    unittest.main()
