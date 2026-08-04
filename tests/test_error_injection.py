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
def test_state_hook_drifts_brain_and_reverts():
    from experiment import mRNN as M
    net = M.MichaelsRNN()

    # delta=0 installs no hook
    ErrorInjector.apply_dist_error(net, delta=0.0)
    assert getattr(net, "_drift_handle", None) is None

    # onset drift: state changes and prev_output stays consistent
    ErrorInjector.apply_dist_error(net, delta=0.2, k=8, seed=1, mode="onset")
    net.reset()
    _ = net(torch.randn(net.num_input_features + 1, 4))
    assert torch.equal(net.prev_output, net.activation_func(net.x))

    # remove restores a clean model
    ErrorInjector.remove_dist_error(net)
    assert getattr(net, "_drift_handle", None) is None
    assert getattr(net, "_drift_original_reset", None) is None
    print("[state-hook] drift installs, stays consistent, and reverts OK")


def test_state_hook_rejects_non_brain_model():
    """apply_dist_error must refuse a model lacking the mRNN state interface."""
    raised = False
    try:
        ErrorInjector.apply_dist_error(nn.Linear(10, 10), delta=0.2)
    except ValueError:
        raised = True
    assert raised, "apply_dist_error should reject a non-mRNN model"
    print("[state-hook] rejects non-brain model OK")


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
