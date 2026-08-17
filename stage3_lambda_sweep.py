#!/usr/bin/env python3
"""
stage3_lambda_sweep.py - sweep the caution weight lambda and ablate the uncertainty estimators.

Takes the M emulators Stage 2 trains, wraps them as an EnsembleLike, and scores the trust-aware
objective across lambda in {0, 0.1, 1, 10, 100} for every estimator that can run and both caution
modes. The output is the table you read to pick a lambda before spending compute on a full run.

It measures the objective DECOMPOSITION, not training-per-lambda: trust_aware_loss is not wired into
cpn_epoch_cpn yet (stage2_ensemble.py flags that as the Stage 3 integration). Lambda only matters if
lam * u is comparable to the task loss, and that is what this shows.

Examples of usage:
  python stage3_lambda_sweep.py --num-members 5 --max-epochs 2000
  python stage3_lambda_sweep.py --num-members 3 --max-epochs 1000 --csv sweep.csv
"""
import argparse
import csv
import sys

import torch

from uncertainty import Estimator, uncertainty
from safety import (SafetyWeights, StimCalibration, safety_cost, soft_penalty,
                    hard_halt, charge_damage_fraction)

import warnings
warnings.filterwarnings("ignore")

LAMBDA_GRID = (0.0, 0.1, 1.0, 10.0, 100.0)   #0 is the no-caution control the rest is read against
CAUTION_MODES = ("soft", "hard")
ESTIMATOR_GRID = (Estimator.DISAGREEMENT, Estimator.MAX_FROBENIUS,
                  Estimator.SINGLE_VAR, Estimator.MC_DROPOUT)
HALT_PENALTY = 10.0                          #MOReL's flat penalty, scaled by lambda


class TrainedENEnsemble(torch.nn.Module):
    #stage 2's trained ENs, dressed up as an EnsembleLike
    #they are single-headed, so forward() reports zero variance. MAX_FROBENIUS and SINGLE_VAR read
    #var, so on these members they come out as 0 -- read those rows as "not measured", not as certain
    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)
        self.M = len(members)

    def forward(self, x_t):
        mu = torch.stack([m(x_t) for m in self.members], dim=0)   #(M, batch, out_dim)
        return mu, torch.zeros_like(mu)

    def summarize(self, x_t):
        mu, var = self.forward(x_t)
        return mu.mean(0), var.mean(0), mu.var(0, unbiased=False)

    def reset(self):
        for m in self.members:
            if hasattr(m, "reset"):
                m.reset()


def run_grid(ensemble, x_t, theta, target, halt_threshold, *, weights=None, calib=None):
    #every (estimator, caution, lambda) cell. One row dict each
    #u only depends on the estimator, so it is computed once per estimator, not once per cell
    weights = weights or SafetyWeights()
    calib = calib or StimCalibration()

    pred = ensemble.forward(x_t)[0].mean(dim=0)
    mse = torch.nn.functional.mse_loss(pred, target).item()
    c_safety = safety_cost(theta, ensemble=ensemble, x_t=x_t,
                           weights=weights, calib=calib).mean().item()
    charge = charge_damage_fraction(theta, calib=calib).mean().item()
    cogan = charge_damage_fraction(theta, calib=calib, mode="cogan").mean().item()

    rows = []
    for est in ESTIMATOR_GRID:
        try:
            u = uncertainty(ensemble, x_t, est)
        except RuntimeError as e:                    #MC_DROPOUT with no dropout layers
            print(f"  SKIP {est.value}: {str(e).splitlines()[0]}")
            continue

        halt_frac = (u > halt_threshold).float().mean().item()
        for caution in CAUTION_MODES:
            for lam in LAMBDA_GRID:
                pen = (soft_penalty(u, lam) if caution == "soft"
                       else hard_halt(u, halt_threshold, lam * HALT_PENALTY)).mean().item()
                rows.append({
                    "estimator": est.value, "caution": caution, "lambda": lam,
                    "mse": mse, "u_mean": u.mean().item(), "penalty": pen,
                    "penalty_over_mse": pen / mse if mse > 0 else float("nan"),
                    "safety_cost": c_safety, "halt_frac": halt_frac,
                    "charge_violation_frac": charge, "cogan_violation_frac": cogan,
                    "total": mse + pen + c_safety,
                })
    return rows


def main():
    import cpn_epoch_cpn
    from stage0_repro import resolve_device, resolve_coadapt
    from stage2_ensemble import train_member

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-members", type=int, default=5)
    p.add_argument("--lesion", choices=["f5m1", "aip", "m1"], default="f5m1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-dir", default="./models")
    p.add_argument("--max-epochs", type=int, default=None, help="Per-member cap. Strongly recommended.")
    p.add_argument("--stop-pct", type=float, default=0.70)
    p.add_argument("--halt-quantile", type=float, default=0.95,
                   help="Quantile of the observed u used as the HALT threshold, so hard mode means "
                        "the same thing for estimators whose raw scales differ.")
    p.add_argument("--probe-batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    #train_member reads a few more stage2 flags off args; use its defaults for them
    args.recover, args.stim_sigma, args.drift_ramp, args.baseline_only = False, 2.175, -1, False

    cpn_epoch_cpn.STOP_PCT, cpn_epoch_cpn.MAX_EPOCHS = args.stop_pct, args.max_epochs
    device, coadapt = resolve_device(args.device), resolve_coadapt("off")

    print("=" * 72)
    print(f"STAGE 3: lambda sweep over {list(LAMBDA_GRID)} x estimator ablation")
    print(f"  M={args.num_members}  lesion={args.lesion}")
    print("=" * 72)

    members = [train_member(i, args.seed + i, "clean", {}, args, device, coadapt, None)[1]
               for i in range(args.num_members)]
    ensemble = TrainedENEnsemble(members)
    ensemble.eval()

    ref = next(ensemble.parameters())
    in_dim = members[0].W.shape[0]
    #probe input. Same caveat stage2 documents for its disagreement number: random rows rank the
    #estimators, they do not calibrate them. Feed captured (obs, stim, trial_end) rows here instead
    x_t = torch.randn(args.probe_batch, in_dim, device=ref.device, dtype=ref.dtype)
    theta = x_t[:, -17:-1]                       #the 16 stim channels, before the trial_end flag

    with torch.no_grad():
        out_dim = ensemble.forward(x_t)[0].shape[-1]
        target = torch.zeros(args.probe_batch, out_dim, device=ref.device, dtype=ref.dtype)
        pooled = torch.cat([uncertainty(ensemble, x_t, Estimator.DISAGREEMENT).flatten()])
        halt_threshold = torch.quantile(pooled, args.halt_quantile).item()
        print(f"\n  halt threshold (q={args.halt_quantile} of observed u): {halt_threshold:.6g}\n")
        rows = run_grid(ensemble, x_t, theta, target, halt_threshold)

    cols = ["estimator", "caution", "lambda", "mse", "u_mean", "penalty",
            "penalty_over_mse", "safety_cost", "halt_frac"]
    print("\n" + "  ".join(c.rjust(12) for c in cols))
    for r in rows:
        print("  ".join((f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])).rjust(12)
                        for c in cols))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  wrote {len(rows)} rows to {args.csv}")

    print("\n  penalty_over_mse ~0 means lambda changes nothing; >>1 means it drowns the task.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
