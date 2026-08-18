"""
Contract tests for the Stage 3 lambda sweep. Uses toy members so it needs no trained model and
never imports the experiment stack.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ensemble_types import EnsembleLike
from uncertainty import Estimator
from stage3_lambda_sweep import (
    LAMBDA_GRID, CAUTION_MODES, ESTIMATOR_GRID, TrainedENEnsemble, run_grid,
)


class _Member(torch.nn.Module):
    """Stands in for a trained single-headed EN: one linear layer, no variance head."""
    def __init__(self, in_dim=37, out_dim=8, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.net(x)


def _setup(M=3, batch=8, in_dim=37, out_dim=8):
    ens = TrainedENEnsemble([_Member(in_dim, out_dim, seed=i) for i in range(M)])
    x = torch.randn(batch, in_dim)
    return ens, x, x[:, -17:-1], torch.zeros(batch, out_dim)


def test_grid_matches_the_deliverable():
    assert LAMBDA_GRID == (0.0, 0.1, 1.0, 10.0, 100.0)
    assert set(CAUTION_MODES) == {"soft", "hard"}
    assert len(ESTIMATOR_GRID) == 4


def test_wrapper_satisfies_protocol():
    ens, x, _, _ = _setup()
    assert isinstance(ens, EnsembleLike) and ens.M == 3
    mu, var = ens.forward(x)
    assert mu.shape == (3, 8, 8) and var.shape == mu.shape
    assert torch.equal(var, torch.zeros_like(var))    # single-headed: no predicted variance


def test_variance_estimators_report_zero_on_single_headed_members():
    # They still run, but var = 0 makes them read as perfectly certain everywhere. Those rows are
    # "not measured", not "no uncertainty": they only mean something once members emit a variance.
    rows = run_grid(*_setup(), halt_threshold=0.5)
    scored = {r["estimator"] for r in rows}
    assert {"disagreement", "max_frobenius", "single_var"} <= scored
    assert "mc_dropout" not in scored              # plain Linear members carry no dropout
    for r in rows:
        if r["estimator"] in ("max_frobenius", "single_var"):
            assert r["u_mean"] == 0.0


def test_lambda_zero_is_the_no_caution_control():
    rows = run_grid(*_setup(), halt_threshold=0.0)
    for r in rows:
        if r["lambda"] == 0.0:
            assert r["penalty"] == 0.0, f"{r['caution']} at lambda=0 must not touch the objective"


def test_penalty_scales_with_lambda():
    rows = [r for r in run_grid(*_setup(), halt_threshold=0.0)
            if r["caution"] == "soft" and r["estimator"] == "disagreement"]
    pens = [r["penalty"] for r in sorted(rows, key=lambda r: r["lambda"])]
    assert pens == sorted(pens)
    assert pens[-1] > pens[1] > 0


def test_grid_covers_every_cell_once():
    rows = run_grid(*_setup(), halt_threshold=0.5)
    assert len(rows) == 3 * len(CAUTION_MODES) * len(LAMBDA_GRID)   # all but mc_dropout
    keys = {(r["estimator"], r["caution"], r["lambda"]) for r in rows}
    assert len(keys) == len(rows)


def test_rows_carry_the_safety_columns():
    row = run_grid(*_setup(), halt_threshold=0.5)[0]
    for col in ("mse", "u_mean", "penalty", "safety_cost", "halt_frac",
                "charge_violation_frac", "cogan_violation_frac", "total"):
        assert col in row, col
    for col in ("halt_frac", "charge_violation_frac", "cogan_violation_frac"):
        assert 0.0 <= row[col] <= 1.0, col          # fractions, not sums


ALL = [
    test_grid_matches_the_deliverable,
    test_wrapper_satisfies_protocol,
    test_variance_estimators_report_zero_on_single_headed_members,
    test_lambda_zero_is_the_no_caution_control,
    test_penalty_scales_with_lambda,
    test_grid_covers_every_cell_once,
    test_rows_carry_the_safety_columns,
]

if __name__ == "__main__":
    for t in ALL:
        t()
        print("OK:", t.__name__)
    print(f"\nSUCCESS: all {len(ALL)} lambda-sweep tests passed.")
