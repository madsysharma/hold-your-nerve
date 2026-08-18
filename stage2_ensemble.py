#!/usr/bin/env python3
"""
stage2_ensemble.py - train an ensemble of M emulator networks (ENs).

Summary:
Where Stage 1 runs one CPN-EN training job under a chosen error axis, Stage 2 builds an ensemble
of M emulators - each trained by the same loop but from a different random seed and
(optionally) a different randomly-drawn error config - then reports each member's
recovery and how much the trained members disagree with each other, which is the raw
epistemic-uncertainty signal the trust-aware objective (Stage 3) consumes.

Why we use per-member training loops:
This repo trains the EN inside the experiment loop (CPNEpochEN, built via ErrorInjector.build_structural_en); there is no standalone
EN trainer. So the honest way to produce M trained emulators without touching the training loop is to run that loop M times. 
Wiring the ensemble in as a single EN (one loop, per-member NLL heads) is the Stage 3 integration (see EnsembleENAdapter);
this script is the Stage 2 deliverable it builds on.


Command line arguments:
--member-errors controls the "random configs (error / no error)" per member:
  none         : every member clean; members differ only by init seed (standard deep ensemble).
  random       : each member independently draws none / parametric / structural with a random magnitude - a diverse ensemble on purpose.
  parametric   : every member gets parametric error at --sigma-p.
  structural   : every member gets the structural knobs (--en-num-neurons / --en-cell-type).
  distributional : every member trained with brain drift at --drift-delta (state hook).

Examples of usage:
  python stage2_ensemble.py --num-members 5 --lesion f5m1 --max-epochs 2000
  python stage2_ensemble.py --num-members 5 --member-errors random --max-epochs 2000
  python stage2_ensemble.py --num-members 3 --member-errors structural --en-cell-type gru --max-epochs 2000
  python stage2_ensemble.py --num-members 5 --baseline-only
"""
import argparse
import os
import random
import sys

import torch

import cpn
import cpn_epoch_cpn
import cpn_epoch_en
import experiment.experiment

from stage0_repro import (
    LESIONS,
    resolve_device,
    resolve_coadapt,
    parse_en_ready_floor,
    _REPO_EN_READY_FLOOR,
)
from stage1_error_injection import build_config_with_error, describe_error

import warnings
warnings.filterwarnings("ignore")

CELL_TYPES = ["lstm", "gru", "rnn"]        # Valid --en-cell-type choices (CLI arg)

# Structural error values only, per the error definition: capacity 87 -> {20,40,160}, topology LSTM -> {GRU, vanilla RNN}.
STRUCT_NEURONS_ERROR = [20, 40, 160]
STRUCT_CELLS_ERROR = ["gru", "rnn"]

# Discrete magnitude sweeps from the error definitions (nonzero levels).
SIGMA_P_LEVELS = [0.05, 0.1, 0.2, 0.4]
DRIFT_DELTA_LEVELS = [0.05, 0.1, 0.2, 0.4]


def draw_member_error(mode, rng, args):
    """Return the error kwargs for one member as (label, kwargs) for build_config_with_error. kwargs use the same names Stage 1 threads."""

    if mode == "none":
        return "clean", {}
    if mode == "parametric":
        return f"parametric sigma_p={args.sigma_p}", {"sigma_p": args.sigma_p}
    if mode == "structural":
        kw = {}
        if args.en_num_neurons is not None:
            kw["en_num_neurons"] = args.en_num_neurons
        if args.en_cell_type is not None:
            kw["en_cell_type"] = args.en_cell_type
        return "structural " + str(kw), kw
    if mode == "distributional":
        return f"distributional drift_delta={args.drift_delta}", {"drift_delta": args.drift_delta}
    if mode == "random":
        # All three error axes plus 'none'. 
        # Note the axes are not the same type of variation: parametric/structural diversify the emulator (weights / architecture), 
        # while distributional drifts the brain during that member's training.
        # Magnitudes/values are drawn from the discrete sets in the error definitions; 
        # structural values exclude the correct defaults so a structural member always carries a real mismatch.
        # (Distributional requires the state-hook wiring.)

        choice = rng.choice(["none", "parametric", "structural", "distributional"])
        if choice == "none":
            return "clean", {}
        if choice == "parametric":
            sp = rng.choice(SIGMA_P_LEVELS)
            return f"parametric sigma_p={sp}", {"sigma_p": sp}
        if choice == "distributional":
            dd = rng.choice(DRIFT_DELTA_LEVELS)
            return f"distributional drift_delta={dd}", {"drift_delta": dd}
        # Structural: perturb either capacity or topology (a single clean mismatch)
        if rng.random() < 0.5:
            neur = rng.choice(STRUCT_NEURONS_ERROR)
            return f"structural n={neur}", {"en_num_neurons": neur}
        cell = rng.choice(STRUCT_CELLS_ERROR)
        return f"structural cell={cell}", {"en_cell_type": cell}
    raise ValueError(f"unknown member-errors mode: {mode}")


def train_member(idx, seed, label, err_kwargs, args, device, coadapt, floor):
    """Build and train one EN member via the standard loop. 
    Returns (max_recovery or None, trained_en).
    """
    torch.manual_seed(seed)  # Different init per ensemble member

    cpn_epoch_en.EN_READY_FLOOR = floor if floor is not None else _REPO_EN_READY_FLOOR

    print("-" * 72)
    print(f"MEMBER {idx + 1}/{args.num_members}  seed={seed}  config: {label}")

    cfg = build_config_with_error(
        args.lesion, device, args.recover, coadapt, args.stim_sigma,
        drift_ramp=args.drift_ramp, **err_kwargs
    )
    coproc = cpn.CPN_EN_CoProc(cfg, log_dir=args.log_dir, recycle_thresh=1)
    exp = experiment.experiment.stage(coproc, cfg)

    if args.baseline_only:
        healthy = exp.comp_loss_healthy.item()
        lesioned = exp.comp_loss_lesioned.item()
        print(f"  baseline: healthy={healthy:.6f}  lesioned={lesioned:.6f}  (no training)")
        return None, coproc.en

    lh = exp.run()
    recov = lh.max_pct_recov
    eta_hat = lh.apparent_at_max_recov
    delta = lh.gap_at_max_recov
    cvar_val = lh.cvar_at_max_recov
    eta_hat_s = f"{eta_hat * 100:.1f}%" if eta_hat is not None else "n/a"
    delta_s = f"{delta * 100:+.1f}pp" if delta is not None else "n/a"
    cvar_s = f"{cvar_val:.4e}" if cvar_val is not None else "n/a"
    print(f"  member {idx + 1}: true eta={recov * 100:.1f}%  apparent eta_hat={eta_hat_s}  "
          f"gap Delta={delta_s}  stim-CVaR={cvar_s}")
    return recov, coproc.en


def ensemble_disagreement(members_ens, cfg_in_dim, batch=8):
    """Feed one common random input to every trained EN and return the mean
    across output dims of the variance across members: a scalar diversity signal. 
    Guarded: returns None if the members' interfaces don't align."""
    try:
        # Build the probe on the SAME device/dtype as the members' weights, else a CUDA-trained EN gets a CPU input and matmul fails!
        ref = next(members_ens[0].parameters())
        x = torch.randn(batch, cfg_in_dim, device=ref.device, dtype=ref.dtype)
        preds = []
        for en in members_ens:
            en.reset() if hasattr(en, "reset") else None
            preds.append(en(x).detach())
        stacked = torch.stack(preds, dim=0)              # (M, batch, out_dim)
        return stacked.var(dim=0, unbiased=False).mean().item()
    except Exception as e:
        print(f"  (disagreement metric skipped: {e})")
        return None


def main():
    p = argparse.ArgumentParser(
        description="Stage 2: train an ensemble of M emulator networks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- Stage 2 knobs ---
    p.add_argument("--num-members", type=int, default=5,
                   help="Number of ensemble members M to train. Default 5.")
    p.add_argument("--member-errors",
                   choices=["none", "random", "parametric", "structural", "distributional"],
                   default="none",
                   help="How each member's error config is chosen. 'random' draws a "
                        "different error (or none) per member; the others apply the same "
                        "config to every member. Default none.")
    # --- Error magnitudes (used per --member-errors) ---
    p.add_argument("--sigma-p", type=float, default=0.1,
                   help="Parametric magnitude for --member-errors parametric.")
    p.add_argument("--en-num-neurons", type=int, default=None,
                   help="Structural EN neuron count for --member-errors structural.")
    p.add_argument("--en-cell-type", choices=CELL_TYPES, default=None,
                   help="Structural EN cell type for --member-errors structural.")
    p.add_argument("--drift-delta", type=float, default=0.2,
                   help="Distributional magnitude for --member-errors distributional.")
    p.add_argument("--drift-ramp", type=int, default=-1,
                   help="Distributional timescale: -1 = slow ramp within one trial (default); 0 = instant onset; N > 0 = slow ramp ACROSS trials over N total steps (multi-session drift).")
    # --- Shared knobs (from stage0/stage1) ---
    p.add_argument("--lesion", choices=["f5m1", "aip", "m1"], default="f5m1",
                   help="Which lesion. (Stage 2 trains one ensemble for one lesion.)")
    p.add_argument("--device", default="cpu",
                   help="'cpu', or a CUDA device index like '0'. Default: cpu.")
    p.add_argument("--log-dir", default="./models")
    p.add_argument("--recover", action="store_true",
                   help="Use the pre-recovered mRNN. f5m1 only.")
    p.add_argument("--coadapt", nargs="?", const="on", default="off",
                   choices=["off", "on", "upstream"], help="Co-adaptation mode. Default off.")
    p.add_argument("--stim-sigma", type=float, default=2.175)
    p.add_argument("--en-ready-floor", default=None,
                   help="EN->CPN readiness floor (bare value or per-lesion pairs).")
    p.add_argument("--stop-pct", type=float, default=0.70,
                   help="Per-member stop threshold. Default 0.70.")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Per-member epoch cap. STRONGLY recommended, especially with "
                        "injected errors, or a member may never reach --stop-pct.")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed; member i uses seed + i for its init/config draw.")
    p.add_argument("--baseline-only", action="store_true",
                   help="Build each member and print baselines, but skip training.")
    args = p.parse_args()

    if args.num_members < 1:
        raise SystemExit("--num-members must be >= 1.")
    if args.member_errors != "none" and args.max_epochs is None and not args.baseline_only:
        print("WARNING: --member-errors " + args.member_errors + " is set but --max-epochs "
              "is not. Injected error can hold true recovery below --stop-pct, so a member "
              "may train indefinitely. Set --max-epochs (e.g. 2000).", flush=True)

    device = resolve_device(args.device)
    os.makedirs(args.log_dir, exist_ok=True)
    floor_per_lesion, floor_fallback = parse_en_ready_floor(args.en_ready_floor)
    floor = floor_per_lesion.get(args.lesion, floor_fallback)
    coadapt = resolve_coadapt(args.coadapt)

    cpn_epoch_cpn.STOP_PCT = args.stop_pct
    cpn_epoch_cpn.MAX_EPOCHS = args.max_epochs

    rng = random.Random(args.seed)

    print("=" * 72)
    print(f"STAGE 2: training an ensemble of M={args.num_members} emulators")
    print(f"  lesion={args.lesion}  member-errors={args.member_errors}  "
          f"device={'cpu' if device is None else 'cuda:' + str(device)}  "
          f"stim_sigma={args.stim_sigma}")
    print("=" * 72)

    members = []           # (label, recovery, trained_en)
    for i in range(args.num_members):
        seed = args.seed + i
        label, err_kwargs = draw_member_error(args.member_errors, rng, args)
        recov, en = train_member(i, seed, label, err_kwargs, args, device, coadapt, floor)
        members.append((label, recov, en))

    # ---- Summary of results ----
    print("\n" + "=" * 72)
    print("STAGE 2 ENSEMBLE SUMMARY")
    print(f"  M={args.num_members}  lesion={args.lesion}  member-errors={args.member_errors}")
    for i, (label, recov, _en) in enumerate(members):
        shown = "n/a" if recov is None else f"{recov * 100:.1f}%"
        print(f"  member {i + 1}: recovery={shown:>7s}   [{label}]")

    recs = [r for _, r, _ in members if r is not None]
    if recs:
        mean_r = sum(recs) / len(recs)
        spread = max(recs) - min(recs)
        print(f"\n  recovery: mean={mean_r * 100:.1f}%  min={min(recs) * 100:.1f}%  "
              f"max={max(recs) * 100:.1f}%  spread={spread * 100:.1f}pp")

        # Diversity of the trained emulators (ie: raw epistemic signal for Stage 3).
        try:
            probe = members[0][2]
            in_dim = probe.W.shape[0] if hasattr(probe, "W") else None
        except Exception:
            in_dim = None
        if in_dim is not None and len(members) > 1:
            dis = ensemble_disagreement([en for _, _, en in members], in_dim)
            if dis is not None:
                print(f"  member prediction disagreement (var across members): {dis:.6e}")
    print("=" * 72)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
