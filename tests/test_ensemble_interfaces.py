"""
Contract tests for the Stage 2/3 interfaces. Uses a fake ensemble so that it needs no trained model.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ensemble_types import EnsembleLike
from uncertainty import Estimator, uncertainty
from safety import stim_energy, soft_penalty, hard_halt, safety_cost, SafetyWeights
from trust_aware import TrustAwareConfig, trust_aware_loss, EnsembleENAdapter


class _FakeEnsemble:
    """Minimal EnsembleLike: M members returning fixed-shape (mu, var)."""
    def __init__(self, M=5, out_dim=8, seed=0):
        self.M = M
        self.out_dim = out_dim
        g = torch.Generator().manual_seed(seed)
        self._g = g

    def forward(self, x_t):
        b = x_t.shape[0]
        mu = torch.randn(self.M, b, self.out_dim, generator=self._g)
        var = torch.rand(self.M, b, self.out_dim, generator=self._g) + 0.1
        return mu, var

    def summarize(self, x_t):
        mu, var = self.forward(x_t)
        return mu.mean(0), var.mean(0), mu.var(0, unbiased=False)

    def reset(self):
        pass


def test_fake_satisfies_protocol():
    assert isinstance(_FakeEnsemble(), EnsembleLike)


def test_uncertainty_shapes_and_signs():
    ens = _FakeEnsemble(M=5, out_dim=8)
    x = torch.randn(4, 37)
    for est in (Estimator.MAX_FROBENIUS, Estimator.DISAGREEMENT):
        u = uncertainty(ens, x, est)
        assert u.shape == (4,)
        assert (u >= 0).all()


def test_unimplemented_estimators_raise():
    ens = _FakeEnsemble()
    x = torch.randn(4, 37)
    for est in (Estimator.MC_DROPOUT, Estimator.SINGLE_VAR):
        try:
            uncertainty(ens, x, est)
            assert False, f"{est} should be NotImplementedError until C fills it in"
        except NotImplementedError:
            pass


def test_safety_energy_and_caution():
    theta = torch.randn(4, 16)
    assert stim_energy(theta).shape == (4,)
    u = torch.tensor([0.1, 2.0, 0.5, 3.0])
    assert torch.allclose(soft_penalty(u, 2.0), 2.0 * u)
    halted = hard_halt(u, threshold=1.0, penalty=10.0)
    assert torch.equal(halted, torch.tensor([0.0, 10.0, 0.0, 10.0]))
    assert safety_cost(theta, weights=SafetyWeights(energy=1.0)).shape == (4,)


def test_trust_aware_loss_assembles():
    pred = torch.randn(4, 8)
    target = torch.randn(4, 8)
    u = torch.rand(4)
    theta = torch.randn(4, 16)
    cfg = TrustAwareConfig(lam=1.0, beta=0.5, caution="soft")
    total, parts = trust_aware_loss(pred, target, u, theta, cfg)
    assert total.dim() == 0
    assert set(parts) == {"mse", "uncertainty", "safety", "total"}
    # hard caution path also assembles
    cfg_hard = TrustAwareConfig(caution="hard", halt_threshold=0.5, halt_penalty=5.0)
    total_h, _ = trust_aware_loss(pred, target, u, theta, cfg_hard)
    assert total_h.dim() == 0


def test_adapter_returns_mu_bar_and_uncertainty():
    ens = _FakeEnsemble(M=5, out_dim=8)
    adapter = EnsembleENAdapter(ens, estimator=Estimator.MAX_FROBENIUS)
    x = torch.randn(4, 37)
    pred = adapter(x)                     # single-EN-style call
    assert pred.shape == (4, 8)           # mu_bar
    u = adapter.last_uncertainty()
    assert u.shape == (4,)


ALL = [
    test_fake_satisfies_protocol,
    test_uncertainty_shapes_and_signs,
    test_unimplemented_estimators_raise,
    test_safety_energy_and_caution,
    test_trust_aware_loss_assembles,
    test_adapter_returns_mu_bar_and_uncertainty,
]

if __name__ == "__main__":
    for t in ALL:
        t()
        print("OK:", t.__name__)
    print(f"\nSUCCESS: all {len(ALL)} interface contract tests passed.")
