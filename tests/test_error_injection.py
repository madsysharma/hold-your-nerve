"""
Comprehensive tests for ErrorInjector: parametric, distributional (standalone
tensor form + state-hook form), and structural error axes. Runs under pytest
(test_* functions) or directly (python tests/test_error_injection.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from error_injection import ErrorInjector, _DistributionalDrift


# ===========================================================================
# PARAMETRIC ERROR
# ===========================================================================
def test_parametric_error_sweep():
    sigma_p_values = [0.0, 0.05, 0.1, 0.2, 0.4, 0.5, 1.0, 1.75, 3.0, 5.0]
    print(f"[parametric] sweep across {len(sigma_p_values)} sigma_p values")
    for sigma_p in sigma_p_values:
        model = nn.Linear(100, 50)
        orig_weight = model.weight.clone().detach()
        orig_std = torch.std(orig_weight)
        ErrorInjector.apply_param_error(model, sigma_p)
        noise_std = torch.std(model.weight - orig_weight)
        expected_std = sigma_p * orig_std
        assert not torch.isnan(model.weight).any()
        assert not torch.isinf(model.weight).any()
        if sigma_p == 0.0:
            assert torch.equal(model.weight, orig_weight)
        else:
            assert not torch.equal(model.weight, orig_weight)
            assert torch.isclose(noise_std, expected_std, rtol=0.15)
    print("[parametric] sweep OK")


def test_parametric_negative_is_noop():
    model = nn.Linear(20, 10)
    w0 = model.weight.clone()
    ErrorInjector.apply_param_error(model, -1.0)
    assert torch.equal(model.weight, w0)
    print("[parametric] negative sigma_p no-op OK")


def test_parametric_reproducible_under_seed():
    def perturb(seed):
        torch.manual_seed(seed)
        m = nn.Linear(40, 30)
        w0 = m.weight.clone()
        ErrorInjector.apply_param_error(m, 0.3)
        return (m.weight - w0).clone()
    assert torch.equal(perturb(123), perturb(123))
    assert not torch.equal(perturb(123), perturb(999))
    print("[parametric] reproducibility under seed OK")


def test_parametric_perturbs_all_weight_tensors():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 8))
    before = {n: p.clone() for n, p in model.named_parameters()}
    ErrorInjector.apply_param_error(model, 0.2)
    for n, p in model.named_parameters():
        if p.numel() > 1:
            assert not torch.equal(p, before[n]), f"param {n} not perturbed"
    print("[parametric] all weight tensors perturbed OK")


# ===========================================================================
# DISTRIBUTIONAL ERROR - standalone tensor form (corrected low-dim orthogonal)
# ===========================================================================
def test_distributional_identity_at_zero():
    x = torch.randn(4, 20)
    assert torch.equal(ErrorInjector.apply_distributional_error(x, 0.0), x)
    print("[distributional] identity at delta=0 OK")


def test_distributional_preserves_shape():
    x2, x1 = torch.randn(8, 30), torch.randn(30)
    assert ErrorInjector.apply_distributional_error(x2, 0.2).shape == x2.shape
    assert ErrorInjector.apply_distributional_error(x1, 0.2).shape == x1.shape
    print("[distributional] shape preserved (2D and 1D) OK")


def test_distributional_seed_reproducible_and_sensitive():
    x = torch.randn(5, 25)
    a = ErrorInjector.apply_distributional_error(x, 0.3, seed=7)
    b = ErrorInjector.apply_distributional_error(x, 0.3, seed=7)
    c = ErrorInjector.apply_distributional_error(x, 0.3, seed=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    print("[distributional] seed reproducibility + sensitivity OK")


def test_distributional_rotation_is_orthogonal():
    """The corrected drift's R is a proper rotation (orthonormal)."""
    drift = _DistributionalDrift(30, delta=0.25, k=8, seed=1)
    err = (drift.R @ drift.R.T - torch.eye(30)).abs().max().item()
    assert err < 1e-4, f"R not orthogonal (max err {err:.2e})"
    print("[distributional] rotation is orthogonal OK")


def test_distributional_is_low_dimensional():
    """The drift is confined to a k-dimensional subspace: rank(R - I) <= 2k."""
    k = 8
    drift = _DistributionalDrift(50, delta=0.3, k=k, seed=2)
    rank = torch.linalg.matrix_rank(drift.R - torch.eye(50)).item()
    assert rank <= 2 * k, f"rank(R - I)={rank} exceeds 2k={2 * k}; not low-dimensional"
    print(f"[distributional] low-dimensional (rank {rank} <= {2 * k}) OK")


def test_distributional_magnitude_increases_with_delta():
    torch.manual_seed(0)
    x = torch.randn(4, 50)
    devs = [
        (ErrorInjector.apply_distributional_error(x, d, seed=11) - x).norm().item()
        for d in (0.0, 0.05, 0.1, 0.2, 0.4)
    ]
    assert devs[0] == 0.0
    assert devs[-1] > devs[1]
    print("[distributional] deviation grows with delta OK")


# ===========================================================================
# DISTRIBUTIONAL ERROR - state-hook form (apply_dist_error on the mRNN)
# ===========================================================================
class _FakeBrain(torch.nn.Module):
    """Minimal stand-in exposing exactly the mRNN interface apply_dist_error needs
    (.x, .prev_output, .activation_func, .num_neurons, .reset). Keeping the
    state-hook unit test on this fake -- rather than the full MichaelsRNN, which
    pulls in numpy tensor-splitting and network-file loading -- makes the test
    portable across Python/numpy/torch versions. The real mRNN+hook integration
    is exercised by stage1_error_injection.py --error distributional."""

    def __init__(self, num_neurons=30):
        super().__init__()
        self.num_neurons = num_neurons
        self.activation_func = torch.nn.Tanh()
        self.x = None
        self.prev_output = None

    def reset(self):
        self.x = None
        self.prev_output = None

    def forward(self, inp):
        # inp: (batch, num_neurons); set the pre-activation state and output.
        self.x = inp
        self.prev_output = self.activation_func(self.x)
        return self.prev_output


def test_state_hook_drifts_brain_and_reverts():
    torch.manual_seed(0)

    def stepped(delta):
        net = _FakeBrain(num_neurons=30)
        if delta > 0.0:
            ErrorInjector.apply_dist_error(net, delta=delta, k=8, seed=1, mode="onset")
        net.reset()
        net(torch.ones(4, net.num_neurons))
        return net

    # delta=0 installs no hook
    net0 = _FakeBrain()
    ErrorInjector.apply_dist_error(net0, delta=0.0)
    assert getattr(net0, "_drift_handle", None) is None

    # onset drift changes the brain state relative to no drift...
    clean = stepped(0.0)
    drifted = stepped(0.2)
    assert not torch.allclose(clean.x, drifted.x), "drift did not change the state"

    # ...and leaves prev_output consistent with the drifted state
    assert torch.allclose(drifted.prev_output, drifted.activation_func(drifted.x))

    # onset fires once per trial: after the first step, no drift is pending
    assert getattr(drifted, "_drift_pending", None) is False
    # reset re-arms it
    drifted.reset()
    assert getattr(drifted, "_drift_pending", None) is True

    # remove restores a clean model
    ErrorInjector.remove_dist_error(drifted)
    assert getattr(drifted, "_drift_handle", None) is None
    assert getattr(drifted, "_drift_original_reset", None) is None
    print("[state-hook] drift installs, changes state, fires once, and reverts OK")


def test_state_hook_rejects_non_brain_model():
    """apply_dist_error must refuse a model lacking the mRNN state interface."""
    raised = False
    try:
        ErrorInjector.apply_dist_error(nn.Linear(10, 10), delta=0.2)
    except ValueError:
        raised = True
    assert raised, "apply_dist_error should reject a non-mRNN model"
    print("[state-hook] rejects non-brain model OK")


def test_state_hook_slow_ramp_is_gradual():
    """mode='slow' ramps the drift gradually to full magnitude over ramp_steps,
    so the per-step deviation grows monotonically and reset re-arms the ramp."""
    torch.manual_seed(0)

    class _FixedBrain(torch.nn.Module):
        # holds a fixed base state each step so we isolate the ramp itself
        def __init__(self, n=30):
            super().__init__()
            self.num_neurons = n
            self.activation_func = torch.nn.Identity()
            self.x = None
            self.prev_output = None
            self._base = torch.ones(4, n)
        def reset(self):
            self.x = None
            self.prev_output = None
        def forward(self, _inp):
            self.x = self._base.clone() if self.x is None else self.x
            self.prev_output = self.activation_func(self.x)
            return self.prev_output

    N = 20
    net = _FixedBrain()
    base = net._base.clone()
    ErrorInjector.apply_dist_error(net, delta=0.2, k=8, seed=1, mode="slow", ramp_steps=N)
    net.reset()
    devs = []
    for _ in range(N):
        net(None)
        devs.append((net.x - base).norm().item())
    assert all(devs[i] <= devs[i + 1] + 1e-6 for i in range(len(devs) - 1)), "ramp not monotonic"
    assert devs[0] < devs[-1], "ramp did not grow"
    # final deviation is close to the full one-shot drift magnitude
    full = _DistributionalDrift(30, 0.2, 8, 1).apply(base)
    assert abs(devs[-1] - (full - base).norm().item()) < 0.1, "ramp did not reach ~full delta"
    # reset restarts the ramp (small deviation again)
    net.reset()
    net(None)
    assert (net.x - base).norm().item() < devs[-1], "reset did not re-arm the ramp"
    # slow mode requires a ramp horizon
    try:
        ErrorInjector.apply_dist_error(_FixedBrain(), delta=0.2, mode="slow")
        assert False, "slow without ramp_steps should raise"
    except ValueError:
        pass
    print("[state-hook] slow ramp is gradual, reaches ~full delta, re-arms on reset OK")


def test_state_hook_multi_session_drift_accumulates():
    """span_trials=True: the session-level drift grows ACROSS trials (each trial
    starts more drifted than the last), reaching full delta over ramp_steps total
    steps and holding -- multi-session MINDFUL drift, distinct from the within-trial
    ramp which restarts every trial."""
    torch.manual_seed(0)

    class _FixedBrain(torch.nn.Module):
        def __init__(self, n=30):
            super().__init__()
            self.num_neurons = n
            self.activation_func = torch.nn.Identity()
            self.x = None
            self.prev_output = None
            self._base = torch.ones(4, n)
        def reset(self):
            self.x = None
            self.prev_output = None
        def forward(self, _inp):
            self.x = self._base.clone() if self.x is None else self.x
            self.prev_output = self.activation_func(self.x)
            return self.prev_output

    trial_len, n_total = 10, 50   # full delta after 50 global steps = 5 trials
    net = _FixedBrain()
    base = net._base.clone()
    ErrorInjector.apply_dist_error(net, delta=0.2, k=8, seed=1, mode="slow",
                                   ramp_steps=n_total, span_trials=True)
    onset = []
    for _ in range(7):
        net.reset()
        net(None)                                  # session drift applied at trial start
        onset.append((net.x - base).norm().item())
        for _ in range(trial_len - 1):
            net(None)
    # grows across the first ~5 trials, then plateaus at full delta
    assert all(onset[i] <= onset[i + 1] + 1e-6 for i in range(5)), "drift did not accumulate"
    assert onset[0] < onset[4], "no growth across sessions"
    full = (_DistributionalDrift(30, 0.2, 8, 1).apply(base) - base).norm().item()
    assert abs(onset[-1] - full) < 0.05, "did not plateau near full delta"
    print("[state-hook] multi-session drift accumulates across trials OK")


# ===========================================================================
# STRUCTURAL ERROR
# ===========================================================================
def test_structural_capacity_mismatch():
    for n in (20, 40, 160):
        en, _ = ErrorInjector.build_structural_en(37, 50, num_neurons=n)
        assert en.num_neurons == n
    en, _ = ErrorInjector.build_structural_en(37, 50)
    assert en.num_neurons == 87
    print("[structural] capacity mismatch OK")


def test_structural_cell_types():
    expected = {"lstm": "StimModelLSTM", "gru": "StimModelGRU", "rnn": "StimModelRNN"}
    for cell, cls_name in expected.items():
        en, _ = ErrorInjector.build_structural_en(37, 50, num_neurons=40, cell_type=cell)
        assert type(en).__name__ == cls_name
    print("[structural] cell types (lstm/gru/rnn) OK")


def test_structural_returns_optimizer():
    en, opt = ErrorInjector.build_structural_en(37, 50, num_neurons=40)
    assert isinstance(en, nn.Module)
    assert isinstance(opt, torch.optim.Optimizer)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert opt_ids == {id(p) for p in en.parameters()}
    print("[structural] returns (model, optimizer) OK")


def test_structural_invalid_cell_type_raises():
    raised = False
    try:
        ErrorInjector.build_structural_en(37, 50, cell_type="bogus")
    except ValueError:
        raised = True
    assert raised
    print("[structural] invalid cell type raises OK")


def test_structural_capacity_changes_param_count():
    small, _ = ErrorInjector.build_structural_en(37, 50, num_neurons=20)
    large, _ = ErrorInjector.build_structural_en(37, 50, num_neurons=160)
    assert sum(p.numel() for p in large.parameters()) > sum(p.numel() for p in small.parameters())
    print("[structural] param count scales with capacity OK")


ALL_TESTS = [
    test_parametric_error_sweep,
    test_parametric_negative_is_noop,
    test_parametric_reproducible_under_seed,
    test_parametric_perturbs_all_weight_tensors,
    test_distributional_identity_at_zero,
    test_distributional_preserves_shape,
    test_distributional_seed_reproducible_and_sensitive,
    test_distributional_rotation_is_orthogonal,
    test_distributional_is_low_dimensional,
    test_distributional_magnitude_increases_with_delta,
    test_state_hook_drifts_brain_and_reverts,
    test_state_hook_rejects_non_brain_model,
    test_state_hook_slow_ramp_is_gradual,
    test_state_hook_multi_session_drift_accumulates,
    test_structural_capacity_mismatch,
    test_structural_cell_types,
    test_structural_returns_optimizer,
    test_structural_invalid_cell_type_raises,
    test_structural_capacity_changes_param_count,
]

if __name__ == "__main__":
    print("=" * 72)
    print("ErrorInjector test suite: parametric / distributional / state-hook / structural")
    print("=" * 72)
    for t in ALL_TESTS:
        t()
    print("\nSUCCESS: all", len(ALL_TESTS), "tests passed.")
