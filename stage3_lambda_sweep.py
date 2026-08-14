#!/usr/bin/env python3
"""
stage3_lambda_sweep.py - sweep the caution weight lambda and ablate the uncertainty estimators.

Stage 2 proves M emulators train and disagree. This script takes those trained members, wraps them
in an EnsembleLike, and sweeps the trust-aware objective across lambda in {0, 0.1, 1, 10, 100} for
every available estimator and both caution modes, reporting how the penalty, the safety cost and the
task error trade off. The output is the table you read to pick a lambda before spending compute on a
full trust-aware training run.

The repo's CPN epoch still optimises the plain task loss - trust_aware_loss is not wired into
cpn_epoch_cpn yet (stage2_ensemble.py flags that wiring as the Stage 3 integration). So this harness
measures the objective DECOMPOSITION on trained members rather than training once per lambda. That is
the useful half: lambda only matters if lam * u is comparable to the task loss, and this is what tells
you which of the five values are even worth training at. Once the loop integration lands, the same
grid drives it - see run_grid(), which takes the per-cell numbers from one function.

The Stage 2 members are single-headed and MSE-trained, so they carry no predicted variance. Anything
reading var - MAX_FROBENIUS, SINGLE_VAR - is unavailable on them and is skipped with a printed reason
rather than silently reporting zeros. DISAGREEMENT always works; MC_DROPOUT works only if the members
carry nn.Dropout layers. Give the members variance heads and the skipped rows appear automatically.

"""
import argparse
import csv
import os
import random
import sys

import torch

# stage0/stage2 pull in the whole experiment stack (cpn, michaels, the mRNN), which takes a while to
# load. They are only needed to TRAIN members, so they are imported inside main() and the grid
# machinery below stays importable on its own.
from uncertainty import Estimator, uncertainty
from safety import (
    SafetyWeights, safety_cost, soft_penalty, hard_halt, halt_mask,
    charge_damage_fraction, ood_fraction, stim_diagnostics,
    StimCalibration, SupportModel,
)

import warnings
warnings.filterwarnings("ignore")

# The lambda grid from the deliverable. 0 is the no-caution control every other cell is read against.
LAMBDA_GRID = (0.0, 0.1, 1.0, 10.0, 100.0)

CAUTION_MODES = ("soft", "hard")

# Estimator ablation. Ordered cheapest-first so a skipped variance estimator does not hide the rest.
ESTIMATOR_GRID = (
    Estimator.DISAGREEMENT,
    Estimator.MAX_FROBENIUS,
    Estimator.SINGLE_VAR,
    Estimator.MC_DROPOUT,
)


class TrainedENEnsemble(torch.nn.Module):
    #the M trained ENs from stage 2, dressed up as an EnsembleLike
    #members are single-headed so there is no predicted variance to hand back. forward() returns
    #zeros for var and sets has_variance=False, and the sweep skips the estimators that would
    #otherwise score every single input as perfectly certain
    #swap in members that emit mean + log-var and the whole estimator grid lights up, nothing here changes

    has_variance = False

    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)
        self.M = len(members)

    def forward(self, x_t):
        mu = torch.stack([m(x_t) for m in self.members], dim=0)   # (M, batch, out_dim)
        return mu, torch.zeros_like(mu)

    def summarize(self, x_t):
        mu, var = self.forward(x_t)
        return mu.mean(0), var.mean(0), mu.var(0, unbiased=False)

    def reset(self):
        for m in self.members:
            if hasattr(m, "reset"):
                m.reset()


def available_estimators(ensemble):
    #which estimators can actually run on this ensemble, and why the rest can't
    usable, skipped = [], []
    has_var = getattr(ensemble, "has_variance", True)
    for est in ESTIMATOR_GRID:
        if est in (Estimator.MAX_FROBENIUS, Estimator.SINGLE_VAR) and not has_var:
            skipped.append((est, "members are single-headed: no predicted variance"))
            continue
        if est == Estimator.MC_DROPOUT:
            # Cheapest way to ask is to try: the estimator itself knows what counts as a dropout layer.
            try:
                uncertainty(ensemble, _probe_like(ensemble), est, n_passes=2)
            except RuntimeError as e:
                skipped.append((est, str(e).split(".")[0]))
                continue
            except Exception as e:                      # a real failure, not "unsupported"
                skipped.append((est, f"errored: {e}"))
                continue
        usable.append(est)
    return usable, skipped


def _probe_like(ensemble, batch=4):
    #throwaway input just to ask an estimator whether it can run at all
    ref = next(ensemble.parameters())
    in_dim = _infer_in_dim(ensemble)
    return torch.randn(batch, in_dim, device=ref.device, dtype=ref.dtype)


def _infer_in_dim(ensemble):
    #how wide is the emulator input. Members don't agree on what they call it, so try the usual names
    first = ensemble.members[0]
    for attr in ("in_dim", "input_size"):
        got = getattr(first, attr, None)
        if got is not None:
            return int(got)
    if hasattr(first, "W"):
        return int(first.W.shape[0])
    raise ValueError(
        "cannot infer the emulator input width; give the member an in_dim attribute."
    )


def score_cell(ensemble, x_t, theta, target, estimator, lam, caution,
               *, halt_threshold, halt_penalty, weights, calib, support):
    #one (estimator, lambda, caution) cell, with the objective broken into its parts
    #same pieces trust_aware.trust_aware_loss assembles, kept separate so we can see which term
    #actually dominates instead of only their sum
    mu, _var = ensemble.forward(x_t)
    pred = mu.mean(dim=0)
    u = uncertainty(ensemble, x_t, estimator)

    mse = torch.nn.functional.mse_loss(pred, target)

    if caution == "soft":
        u_term = soft_penalty(u, lam).mean()
        halted = halt_mask(u, halt_threshold)
    else:
        # MOReL scales its flat penalty by lambda too, so lambda = 0 is genuinely no-caution in
        # both modes and the two columns stay comparable.
        u_term = hard_halt(u, halt_threshold, lam * halt_penalty).mean()
        halted = halt_mask(u, halt_threshold)

    c_safety = safety_cost(theta, ensemble=ensemble, x_t=x_t, weights=weights,
                           calib=calib, support=support).mean()

    return {
        "estimator": estimator.value,
        "caution": caution,
        "lambda": lam,
        "mse": mse.item(),
        "u_mean": u.mean().item(),
        "u_max": u.max().item(),
        "penalty": u_term.item(),
        "penalty_over_mse": (u_term / mse).item() if mse.item() > 0 else float("nan"),
        "safety_cost": c_safety.item(),
        "halt_frac": halted.float().mean().item(),
        "charge_violation_frac": charge_damage_fraction(theta, calib=calib).mean().item(),
        "cogan_violation_frac": charge_damage_fraction(theta, calib=calib, mode="cogan").mean().item(),
        "ood_frac": ood_fraction(ensemble, x_t, width_threshold=1.0, support=support).mean().item(),
        "total": (mse + u_term + weights.energy * c_safety).item(),
    }


def run_grid(ensemble, x_t, theta, target, *, estimators, halt_threshold, halt_penalty,
             weights, calib, support):
    #every (estimator, caution, lambda) cell. One row dict each
    rows = []
    for est in estimators:
        for caution in CAUTION_MODES:
            for lam in LAMBDA_GRID:
                rows.append(score_cell(
                    ensemble, x_t, theta, target, est, lam, caution,
                    halt_threshold=halt_threshold, halt_penalty=halt_penalty,
                    weights=weights, calib=calib, support=support,
                ))
    return rows


def print_table(rows):
    #the columns worth eyeballing in the terminal. The CSV keeps everything
    cols =["estimator", "caution", "lambda", "mse", "u_mean", "penalty",
            "penalty_over_mse", "safety_cost", "halt_frac", "ood_frac"]
    widths = [max(len(c), 12) for c in cols]
    print("  ".join(c.rjust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        cells = []
        for c, w in zip(cols, widths):
            v = r[c]
            cells.append((f"{v:.4g}" if isinstance(v, float) else str(v)).rjust(w))
        print("  ".join(cells))


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  wrote {len(rows)} rows to {path}")


def main():
    import cpn_epoch_cpn
    from stage0_repro import resolve_device, resolve_coadapt, parse_en_ready_floor
    from stage2_ensemble import CELL_TYPES, draw_member_error, train_member

    p = argparse.ArgumentParser(
        description="Stage 3: sweep lambda and ablate the uncertainty estimators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- Stage 3 knobs ---
    p.add_argument("--halt-threshold", type=float, default=None,
                   help="MOReL HALT threshold on u. Default: the --halt-quantile quantile of the u "
                        "actually observed, so the threshold means the same thing for every estimator.")
    p.add_argument("--halt-quantile", type=float, default=0.95,
                   help="Quantile of observed u used when --halt-threshold is not given. Default 0.95.")
    p.add_argument("--halt-penalty", type=float, default=10.0,
                   help="MOReL penalty applied where u exceeds the threshold, scaled by lambda.")
    p.add_argument("--energy-weight", type=float, default=1.0)
    p.add_argument("--charge-weight", type=float, default=0.0,
                   help="Weight on the Shannon damage fraction. The fractions live in [0,1] while "
                        "energy does not, so this has to be large to matter.")
    p.add_argument("--ood-weight", type=float, default=0.0)
    p.add_argument("--probe-batch", type=int, default=64,
                   help="Rows in the probe batch the grid is scored on.")
    p.add_argument("--csv", default=None, help="Write the grid to this CSV.")
    # --- Ensemble knobs, same meanings as Stage 2 ---
    p.add_argument("--num-members", type=int, default=5)
    p.add_argument("--member-errors",
                   choices=["none", "random", "parametric", "structural", "distributional"],
                   default="none")
    p.add_argument("--sigma-p", type=float, default=0.1)
    p.add_argument("--en-num-neurons", type=int, default=None)
    p.add_argument("--en-cell-type", choices=CELL_TYPES, default=None)
    p.add_argument("--drift-delta", type=float, default=0.2)
    p.add_argument("--drift-ramp", type=int, default=-1)
    # --- Shared knobs from stage0/stage1/stage2 ---
    p.add_argument("--lesion", choices=["f5m1", "aip", "m1"], default="f5m1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-dir", default="./models")
    p.add_argument("--recover", action="store_true")
    p.add_argument("--coadapt", nargs="?", const="on", default="off",
                   choices=["off", "on", "upstream"])
    p.add_argument("--stim-sigma", type=float, default=2.175)
    p.add_argument("--en-ready-floor", default=None)
    p.add_argument("--stop-pct", type=float, default=0.70)
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Per-member epoch cap. STRONGLY recommended.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline-only", action="store_true",
                   help="Build the members without training them. Fast way to check the grid runs.")
    args = p.parse_args()

    if args.num_members < 2:
        raise SystemExit("--num-members must be >= 2: every estimator needs members to compare.")
    if args.max_epochs is None and not args.baseline_only:
        print("WARNING: --max-epochs is not set; each of the "
              f"{args.num_members} members trains to --stop-pct. This can take a long time.",
              flush=True)

    device = resolve_device(args.device)
    os.makedirs(args.log_dir, exist_ok=True)
    floor_per_lesion, floor_fallback = parse_en_ready_floor(args.en_ready_floor)
    floor = floor_per_lesion.get(args.lesion, floor_fallback)
    coadapt = resolve_coadapt(args.coadapt)

    cpn_epoch_cpn.STOP_PCT = args.stop_pct
    cpn_epoch_cpn.MAX_EPOCHS = args.max_epochs

    rng = random.Random(args.seed)

    print("=" * 72)
    print(f"STAGE 3: lambda sweep over {list(LAMBDA_GRID)} x estimator ablation")
    print(f"  M={args.num_members}  lesion={args.lesion}  member-errors={args.member_errors}  "
          f"device={'cpu' if device is None else 'cuda:' + str(device)}")
    print("=" * 72)

    members = []
    for i in range(args.num_members):
        label, err_kwargs = draw_member_error(args.member_errors, rng, args)
        _recov, en = train_member(i, args.seed + i, label, err_kwargs, args,
                                  device, coadapt, floor)
        members.append(en)

    ensemble = TrainedENEnsemble(members)
    ensemble.eval()

    usable, skipped = available_estimators(ensemble)
    print("\n" + "-" * 72)
    for est, why in skipped:
        print(f"  SKIP {est.value}: {why}")
    if not usable:
        raise SystemExit("no estimator can run against this ensemble; nothing to sweep.")
    print(f"  estimators in the ablation: {[e.value for e in usable]}")

    # Probe batch. Same caveat stage2_ensemble.py documents for its disagreement number: these are
    # random inputs, not real rollout states, so the numbers rank the estimators rather than
    # calibrate them. Feed captured (obs, stim, trial_end) rows here once the loop hook exists.
    in_dim = _infer_in_dim(ensemble)
    ref = next(ensemble.parameters())
    x_t = torch.randn(args.probe_batch, in_dim, device=ref.device, dtype=ref.dtype)
    with torch.no_grad():
        mu, _ = ensemble.forward(x_t)
    out_dim = mu.shape[-1]
    target = torch.zeros(args.probe_batch, out_dim, device=ref.device, dtype=ref.dtype)

    # theta is the stim block of the emulator input: everything between the observations and the
    # trial-end flag. Without the real split we take the trailing columns minus the flag.
    n_channels = min(16, in_dim - 1)
    theta = x_t[:, -(n_channels + 1):-1]

    calib = StimCalibration()
    weights = SafetyWeights(energy=args.energy_weight,
                            charge=args.charge_weight,
                            ood=args.ood_weight)

    # One threshold for the whole grid, measured from the u this ensemble actually produces, so hard
    # mode is comparable across estimators whose raw scales differ by orders of magnitude.
    if args.halt_threshold is not None:
        halt_threshold = args.halt_threshold
    else:
        with torch.no_grad():
            pooled = torch.cat([uncertainty(ensemble, x_t, e).flatten() for e in usable])
        halt_threshold = torch.quantile(pooled, args.halt_quantile).item()
        print(f"  halt threshold (q={args.halt_quantile} of observed u): {halt_threshold:.6g}")

    diag = stim_diagnostics(theta, calib)
    print(f"  stim: amp_max={diag['amp_max']:.4g}  amp_p99={diag['amp_p99']:.4g}  "
          f"shannon_k_max={diag['shannon_k_max']:.4g}")
    print(f"  (calibrate with StimCalibration.calibrate_ua_per_unit(reference_amp="
          f"{diag['amp_p99']:.4g}) to put the k threshold on this run's scale)")
    print("-" * 72 + "\n")

    with torch.no_grad():
        rows = run_grid(ensemble, x_t, theta, target,
                        estimators=usable, halt_threshold=halt_threshold,
                        halt_penalty=args.halt_penalty, weights=weights,
                        calib=calib, support=None)

    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)

    print("\n" + "=" * 72)
    print("  Read penalty_over_mse: lambda values where it is ~0 leave the objective unchanged,")
    print("  and values where it dwarfs 1 drown the task. The usable band is in between.")
    print("=" * 72)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
