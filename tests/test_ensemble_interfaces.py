"""
Contract tests for the Stage 2/3 interfaces. Uses a fake ensemble so that it needs no trained model.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ensemble_types import EnsembleLike
from uncertainty import Estimator, uncertainty, calibration_curve
from safety import (
    stim_energy, soft_penalty, hard_halt, safety_cost, SafetyWeights,
    charge_damage_fraction, ood_fraction, halt_mask,
    StimCalibration, SupportModel, SHANNON_K_THRESHOLD,
)
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


class _DropoutEnsemble(torch.nn.Module):
    """EnsembleLike whose members carry real nn.Dropout, so MC_DROPOUT has
    something to switch on at inference."""
    def __init__(self, M=3, in_dim=37, out_dim=8, p=0.5):
        super().__init__()
        self.M = M
        self.members = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim), torch.nn.Dropout(p))
            for _ in range(M)
        )

    def forward(self, x_t):
        mu = torch.stack([m(x_t) for m in self.members], dim=0)
        return mu, torch.ones_like(mu)

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
    for est in (Estimator.MAX_FROBENIUS, Estimator.DISAGREEMENT, Estimator.SINGLE_VAR):
        u = uncertainty(ens, x, est)
        assert u.shape == (4,)
        assert (u >= 0).all()


def test_single_var_is_first_members_variance():
    ens = _FakeEnsemble(M=5, out_dim=8, seed=3)
    x = torch.randn(4, 37)
    # Reproduce the same draw the estimator will see, then check the reduction.
    ref = _FakeEnsemble(M=5, out_dim=8, seed=3)
    _mu, var = ref.forward(x)
    assert torch.allclose(uncertainty(ens, x, Estimator.SINGLE_VAR), var[0].sum(dim=-1))


def test_mc_dropout_needs_dropout_layers():
    ens = _FakeEnsemble()          # no nn.Dropout anywhere
    x = torch.randn(4, 37)
    try:
        uncertainty(ens, x, Estimator.MC_DROPOUT)
        assert False, "MC_DROPOUT must refuse an ensemble with no dropout layers"
    except RuntimeError:
        pass


def test_mc_dropout_varies_and_restores_mode():
    ens = _DropoutEnsemble(M=3, in_dim=37, out_dim=8, p=0.5)
    ens.eval()                     # dropout starts OFF, as at inference
    x = torch.randn(4, 37)

    u = uncertainty(ens, x, Estimator.MC_DROPOUT, n_passes=8)
    assert u.shape == (4,)
    assert (u >= 0).all()
    assert (u > 0).any(), "dropout should make the passes differ"

    # The estimator must leave the ensemble exactly as it found it.
    drops = [m for m in ens.modules() if isinstance(m, torch.nn.Dropout)]
    assert drops and not any(m.training for m in drops)


def test_mc_dropout_rejects_one_pass():
    ens = _DropoutEnsemble()
    x = torch.randn(4, 37)
    try:
        uncertainty(ens, x, Estimator.MC_DROPOUT, n_passes=1)
        assert False, "one pass has no variance"
    except ValueError:
        pass


def test_safety_energy_and_caution():
    theta = torch.randn(4, 16)
    assert stim_energy(theta).shape == (4,)
    u = torch.tensor([0.1, 2.0, 0.5, 3.0])
    assert torch.allclose(soft_penalty(u, 2.0), 2.0 * u)
    halted = hard_halt(u, threshold=1.0, penalty=10.0)
    assert torch.equal(halted, torch.tensor([0.0, 10.0, 0.0, 10.0]))
    assert safety_cost(theta, weights=SafetyWeights(energy=1.0)).shape == (4,)


def test_charge_damage_fraction_shannon():
    calib = StimCalibration()
    amp = calib.amp_at_k_threshold          # exactly on the k = 1.85 line
    theta = torch.tensor([[0.5 * amp, 0.5 * amp],     # both channels under
                          [2.0 * amp, 0.0]])          # one channel over -> step counts
    frac = charge_damage_fraction(theta)
    assert frac.shape == (2,)
    assert torch.equal(frac, torch.tensor([0.0, 1.0]))


def test_charge_damage_fraction_over_time():
    calib = StimCalibration()
    amp = calib.amp_at_k_threshold
    theta = torch.tensor([[[2.0 * amp], [0.0]]])      # (batch=1, T=2, C=1)
    assert torch.equal(charge_damage_fraction(theta), torch.tensor([0.5]))


def test_cogan_is_stricter_than_shannon():
    calib = StimCalibration()
    # An amplitude between the Cogan ceiling and the Shannon line: the flat
    # charge cap catches it, the charge-density criterion does not.
    amp = 0.5 * (calib.amp_at_cogan_limit + calib.amp_at_k_threshold)
    assert calib.amp_at_cogan_limit < calib.amp_at_k_threshold
    theta = torch.tensor([[amp]])
    assert torch.equal(charge_damage_fraction(theta, mode="shannon"), torch.tensor([0.0]))
    assert torch.equal(charge_damage_fraction(theta, mode="cogan"), torch.tensor([1.0]))


def test_charge_damage_rejects_unknown_mode():
    try:
        charge_damage_fraction(torch.zeros(2, 3), mode="cogen")
        assert False, "typo'd mode should not silently fall through"
    except ValueError:
        pass


def test_ood_fraction_from_predictive_width():
    ens = _FakeEnsemble(M=5, out_dim=8)
    x = torch.randn(4, 37)
    assert torch.equal(ood_fraction(ens, x, width_threshold=0.0), torch.ones(4))
    assert torch.equal(ood_fraction(ens, x, width_threshold=1e9), torch.zeros(4))


def test_ood_fraction_over_time():
    ens = _FakeEnsemble(M=5, out_dim=8)
    x = torch.randn(4, 6, 37)          # (batch, T, in_dim)
    frac = ood_fraction(ens, x, width_threshold=0.0)
    assert frac.shape == (4,)
    assert torch.equal(frac, torch.ones(4))


def test_support_model_unfitted_flags_nothing():
    sm = SupportModel(dim=6)
    x = torch.randn(4, 6)
    assert not sm.is_fitted
    assert torch.equal(ood_fraction(None, x, width_threshold=1.0, support=sm), torch.zeros(4))


def test_support_model_fits_and_scores():
    torch.manual_seed(0)
    sm = SupportModel(dim=6, quantile=0.95)
    for _ in range(5):
        sm.observe(torch.randn(600, 6))     # 3000 rows > min_samples
    assert sm.fit()
    assert sm.is_fitted and sm.threshold is not None

    frac = ood_fraction(None, torch.randn(64, 6), width_threshold=1.0, support=sm)
    assert frac.shape == (64,)
    assert ((frac >= 0) & (frac <= 1)).all()

    # Far outside the training support: everything should trip.
    far = ood_fraction(None, torch.randn(64, 6) + 50.0, width_threshold=1.0, support=sm)
    assert torch.equal(far, torch.ones(64))


def test_safety_cost_ood_needs_inputs():
    theta = torch.randn(4, 16)
    try:
        safety_cost(theta, weights=SafetyWeights(energy=1.0, ood=1.0))
        assert False, "ood weight without x_t should fail loudly"
    except ValueError:
        pass


def test_safety_cost_combines_all_three():
    ens = _FakeEnsemble(M=5, out_dim=8)
    theta = torch.randn(4, 16)
    x = torch.randn(4, 37)
    cost = safety_cost(theta, ensemble=ens, x_t=x,
                       weights=SafetyWeights(energy=1.0, charge=1.0, ood=1.0),
                       width_threshold=0.0)
    assert cost.shape == (4,)
    assert (cost >= stim_energy(theta)).all()   # the two fractions only add


def test_halt_mask_latches():
    u1 = torch.tensor([0.0, 2.0, 0.0, 0.0])
    u2 = torch.tensor([0.0, 0.0, 3.0, 0.0])
    m1 = halt_mask(u1, threshold=1.0)
    m2 = halt_mask(u2, threshold=1.0, latch=m1)
    assert torch.equal(m1, torch.tensor([False, True, False, False]))
    # Trial 1 dropped back under the threshold but stays halted.
    assert torch.equal(m2, torch.tensor([False, True, True, False]))


def test_calibration_curve_shapes():
    ens = _FakeEnsemble(M=5, out_dim=8)
    loader = [(torch.randn(32, 37), torch.randn(32, 8)) for _ in range(4)]
    centers, errors = calibration_curve(ens, loader, "structural", n_bins=5)
    assert centers.dim() == 1 and errors.dim() == 1
    assert centers.shape == errors.shape
    assert centers.numel() <= 5
    # Bin centres come out ordered, since the bins are quantiles of u.
    assert torch.all(centers[1:] >= centers[:-1])


def test_calibration_curve_rejects_bad_axis_and_mc_dropout():
    ens = _FakeEnsemble()
    loader = [(torch.randn(8, 37), torch.randn(8, 8))]
    for kwargs, axis in ((dict(), "typo"), (dict(estimator=Estimator.MC_DROPOUT), "structural")):
        try:
            calibration_curve(ens, loader, axis, **kwargs)
            assert False, "should reject"
        except ValueError:
            pass


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
    test_single_var_is_first_members_variance,
    test_mc_dropout_needs_dropout_layers,
    test_mc_dropout_varies_and_restores_mode,
    test_mc_dropout_rejects_one_pass,
    test_safety_energy_and_caution,
    test_charge_damage_fraction_shannon,
    test_charge_damage_fraction_over_time,
    test_cogan_is_stricter_than_shannon,
    test_charge_damage_rejects_unknown_mode,
    test_ood_fraction_from_predictive_width,
    test_ood_fraction_over_time,
    test_support_model_unfitted_flags_nothing,
    test_support_model_fits_and_scores,
    test_safety_cost_ood_needs_inputs,
    test_safety_cost_combines_all_three,
    test_halt_mask_latches,
    test_calibration_curve_shapes,
    test_calibration_curve_rejects_bad_axis_and_mc_dropout,
    test_trust_aware_loss_assembles,
    test_adapter_returns_mu_bar_and_uncertainty,
]

if __name__ == "__main__":
    for t in ALL:
        t()
        print("OK:", t.__name__)
    print(f"\nSUCCESS: all {len(ALL)} interface contract tests passed.")
