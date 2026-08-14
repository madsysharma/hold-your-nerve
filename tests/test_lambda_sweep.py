"""
Contract tests for the Stage 3 lambda sweep. Uses toy ensembles so it needs no trained model
and never imports the experiment stack.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ensemble_types import EnsembleLike
from uncertainty import Estimator
from safety import SafetyWeights, StimCalibration
from stage3_lambda_sweep import (
    LAMBDA_GRID, CAUTION_MODES, ESTIMATOR_GRID,
    TrainedENEnsemble, available_estimators, score_cell, run_grid, write_csv,
)


class _Member(torch.nn.Module):
    """Stands in for a trained single-headed EN: one linear layer, no variance."""
    def __init__(self, in_dim=37, out_dim=8, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.in_dim = in_dim
        self.net = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.net(x)


class _DropoutMember(_Member):
    def __init__(self, in_dim=37, out_dim=8, seed=0, p=0.5):
        super().__init__(in_dim, out_dim, seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim), torch.nn.Dropout(p))


def _ens(cls=_Member, M=3):
    return TrainedENEnsemble([cls(seed=i) for i in range(M)])


def _batch(ens, batch=8, in_dim=37, out_dim=8):
    x = torch.randn(batch, in_dim)
    theta = x[:, -17:-1]
    target = torch.zeros(batch, out_dim)
    return x, theta, target


def test_grid_matches_the_deliverable():
    assert LAMBDA_GRID == (0.0, 0.1, 1.0, 10.0, 100.0)
    assert set(CAUTION_MODES) == {"soft", "hard"}
    assert len(ESTIMATOR_GRID) == 4


def test_wrapper_satisfies_protocol():
    ens = _ens()
    assert isinstance(ens, EnsembleLike)
    assert ens.M == 3
    mu, var = ens.forward(torch.randn(8, 37))
    assert mu.shape == (3, 8, 8)
    assert var.shape == mu.shape
    assert torch.equal(var, torch.zeros_like(var))   # single-headed: no predicted variance
    assert ens.has_variance is False


def test_variance_estimators_are_skipped_not_faked():
    usable, skipped = available_estimators(_ens())
    names = {e.value for e in usable}
    skipped_names = {e.value for e, _why in skipped}
    assert "disagreement" in names
    # Reporting u = 0 everywhere would read as "perfectly certain" and quietly poison the sweep.
    assert {"max_frobenius", "single_var"} <= skipped_names
    assert "mc_dropout" in skipped_names          # plain Linear members have no dropout
    assert all(why for _e, why in skipped)        # every skip explains itself


def test_dropout_members_unlock_mc_dropout():
    usable, _skipped = available_estimators(_ens(_DropoutMember))
    assert Estimator.MC_DROPOUT in usable


def test_lambda_zero_is_the_no_caution_control():
    ens = _ens()
    x, theta, target = _batch(ens)
    kw = dict(halt_threshold=1e9, halt_penalty=10.0,
              weights=SafetyWeights(energy=1.0), calib=StimCalibration(), support=None)
    for caution in CAUTION_MODES:
        row = score_cell(ens, x, theta, target, Estimator.DISAGREEMENT, 0.0, caution, **kw)
        assert row["penalty"] == 0.0, f"{caution} at lambda=0 must not touch the objective"


def test_penalty_scales_with_lambda():
    ens = _ens()
    x, theta, target = _batch(ens)
    kw = dict(halt_threshold=0.0, halt_penalty=10.0,
              weights=SafetyWeights(energy=1.0), calib=StimCalibration(), support=None)
    pens = [score_cell(ens, x, theta, target, Estimator.DISAGREEMENT, lam, "soft", **kw)["penalty"]
            for lam in LAMBDA_GRID]
    assert pens == sorted(pens)
    assert pens[-1] > pens[1] > 0


def test_run_grid_covers_every_cell():
    ens = _ens()
    x, theta, target = _batch(ens)
    usable, _ = available_estimators(ens)
    rows = run_grid(ens, x, theta, target, estimators=usable,
                    halt_threshold=0.5, halt_penalty=10.0,
                    weights=SafetyWeights(energy=1.0), calib=StimCalibration(), support=None)
    assert len(rows) == len(usable) * len(CAUTION_MODES) * len(LAMBDA_GRID)
    keys = {(r["estimator"], r["caution"], r["lambda"]) for r in rows}
    assert len(keys) == len(rows)                 # no duplicated cells


def test_rows_carry_the_safety_columns():
    ens = _ens()
    x, theta, target = _batch(ens)
    row = score_cell(ens, x, theta, target, Estimator.DISAGREEMENT, 1.0, "soft",
                     halt_threshold=0.5, halt_penalty=10.0,
                     weights=SafetyWeights(energy=1.0), calib=StimCalibration(), support=None)
    for col in ("mse", "u_mean", "penalty", "safety_cost", "halt_frac",
                "charge_violation_frac", "cogan_violation_frac", "ood_frac", "total"):
        assert col in row, col
    for col in ("halt_frac", "charge_violation_frac", "cogan_violation_frac", "ood_frac"):
        assert 0.0 <= row[col] <= 1.0, col        # fractions, not sums


def test_csv_round_trips(tmp_path=None):
    import csv as _csv
    import tempfile
    ens = _ens()
    x, theta, target = _batch(ens)
    usable, _ = available_estimators(ens)
    rows = run_grid(ens, x, theta, target, estimators=usable,
                    halt_threshold=0.5, halt_penalty=10.0,
                    weights=SafetyWeights(energy=1.0), calib=StimCalibration(), support=None)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sweep.csv")
        write_csv(rows, path)
        with open(path, newline="") as f:
            back = list(_csv.DictReader(f))
    assert len(back) == len(rows)
    assert back[0]["estimator"] == rows[0]["estimator"]


ALL = [
    test_grid_matches_the_deliverable,
    test_wrapper_satisfies_protocol,
    test_variance_estimators_are_skipped_not_faked,
    test_dropout_members_unlock_mc_dropout,
    test_lambda_zero_is_the_no_caution_control,
    test_penalty_scales_with_lambda,
    test_run_grid_covers_every_cell,
    test_rows_carry_the_safety_columns,
    test_csv_round_trips,
]

if __name__ == "__main__":
    for t in ALL:
        t()
        print("OK:", t.__name__)
    print(f"\nSUCCESS: all {len(ALL)} lambda-sweep tests passed.")
