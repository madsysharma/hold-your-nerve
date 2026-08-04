#!/usr/bin/env python3
"""
stage1_error_injection.py: run CPN-EN training with a chosen emulator-error axis.

This is the Stage 1 companion to stage0_repro.py. 
Where Stage 0 reproduces the clean testbed, Stage 1 injects a controlled emulator error and runs the same CPN-EN training so you can see how recovery degrades. 
Mirrors stage0_repro's CLI and reuses its helpers.

The three benchmark axes and how they are integrated into the training process:

  parametric     : cfg.sigma_p -> ErrorInjector.apply_param_error on the EN at the EN->CPN handoff (cpn.py).
  structural     : cfg.en_num_neurons / cfg.en_cell_type -> ErrorInjector.build_structural_en builds the EN with a mismatched capacity or cell type 
                   (cpn_epoch_en.py).
  distributional : cfg.drift_delta -> ErrorInjector.apply_dist_error installs a forward hook on the brain (self.mike) at the EN->CPN handoff (experiment.py),
                   drifting the brain state x->Rx+b while the EN is frozen stale (cpn.py).

Examples of usage:
  python stage1_error_injection.py --error parametric --sigma-p 0.2 --lesion f5m1
  python stage1_error_injection.py --error structural --en-num-neurons 40 --lesion m1
  python stage1_error_injection.py --error structural --en-cell-type gru --lesion aip
  python stage1_error_injection.py --error distributional --drift-delta 0.2 --lesion f5m1
  python stage1_error_injection.py --error none --baseline-only
"""
import argparse
import os
import sys

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

import warnings
warnings.filterwarnings("ignore")


def build_config_with_error(lesion_key, device, recover, coadapt, stim_sigma,
                            sigma_p=0.0, en_num_neurons=None, en_cell_type=None,
                            drift_delta=0.0, drift_k=8, drift_seed=42):
    """Like stage0_repro.build_config, but also threads the error-injection knobs (parametric sigma_p; structural en_num_neurons/en_cell_type; distributional
    drift_delta/drift_k/drift_seed) into the repo Config.

    Unset knobs are omitted so the repo defaults stand."""
    common = dict(
        recover_after_lesion=recover,
        cuda=device,
        stim_sigma=stim_sigma,
        sigma_p=sigma_p,
        drift_delta=drift_delta,
        drift_k=drift_k,
        drift_seed=drift_seed,
    )
    if coadapt is not None:
        common["coadapt"] = coadapt
    if en_num_neurons is not None:
        common["en_num_neurons"] = en_num_neurons
    if en_cell_type is not None:
        common["en_cell_type"] = en_cell_type

    if lesion_key == "f5m1":
        return experiment.experiment.get_config(**common)
    if lesion_key == "aip":
        return experiment.experiment.get_aip_lesion_config(**common)
    if lesion_key == "m1":
        return experiment.experiment.get_m1_lesion_config(**common)
    raise ValueError(f"Unknown lesion key: {lesion_key}")


def describe_error(error, sigma_p, en_num_neurons, en_cell_type, drift_delta):
    if error == "none":
        return "none (clean baseline)"
    if error == "parametric":
        return f"parametric: sigma_p={sigma_p}"
    if error == "distributional":
        return f"distributional (state hook): drift_delta={drift_delta}"
    if error == "structural":
        parts = []
        if en_num_neurons is not None:
            parts.append(f"en_num_neurons={en_num_neurons}")
        if en_cell_type is not None:
            parts.append(f"en_cell_type={en_cell_type}")
        return "structural: " + (", ".join(parts) if parts else "(no knob set)")
    return error


def run_one(lesion_key, device, log_dir, recover, coadapt, stim_sigma,
            en_ready_floor, baseline_only, error, sigma_p,
            en_num_neurons, en_cell_type, drift_delta, drift_k, drift_seed):
    desc, recover_supported = LESIONS[lesion_key]
    if recover and not recover_supported:
        raise SystemExit(
            f"--recover is not available for the {lesion_key} lesion: no pre-recovered "
            f"mRNN model is bundled for it. Only the f5m1 lesion supports --recover."
        )

    cpn_epoch_en.EN_READY_FLOOR = (
        en_ready_floor if en_ready_floor is not None else _REPO_EN_READY_FLOOR
    )

    label = describe_error(error, sigma_p, en_num_neurons, en_cell_type, drift_delta)
    print("=" * 72)
    print(f"LESION: {lesion_key}  ->  {desc}")
    print(f"  ERROR: {label}")
    print(f"  device={'cpu' if device is None else 'cuda:' + str(device)}  "
          f"recover_after_lesion={recover}  "
          f"coadapt={'upstream default' if coadapt is None else coadapt}  "
          f"stim_sigma={stim_sigma}")
    print(f"  en_ready_floor={cpn_epoch_en.EN_READY_FLOOR}  "
          f"stop_pct={cpn_epoch_cpn.STOP_PCT}  max_epochs={cpn_epoch_cpn.MAX_EPOCHS}")
    print("=" * 72)

    cfg = build_config_with_error(
        lesion_key, device, recover, coadapt, stim_sigma,
        sigma_p=sigma_p, en_num_neurons=en_num_neurons, en_cell_type=en_cell_type,
        drift_delta=drift_delta, drift_k=drift_k, drift_seed=drift_seed,
    )

    coproc = cpn.CPN_EN_CoProc(cfg, log_dir=log_dir, recycle_thresh=1)
    exp = experiment.experiment.stage(coproc, cfg)

    healthy = exp.comp_loss_healthy.item()
    lesioned = exp.comp_loss_lesioned.item()
    print(f"\n  Healthy ceiling  (MSE, no lesion)          : {healthy:.6f}")
    print(f"  Lesioned floor   (MSE, lesion, no stim)    : {lesioned:.6f}")
    print(f"  Headroom (lesioned - healthy)              : {lesioned - healthy:.6f}")
    if lesioned <= healthy:
        print("\n  WARNING: lesioned loss is not worse than healthy; recovery will be "
              "meaningless. Check the lesion selection.")

    if baseline_only:
        print("\n  --baseline-only set: skipping CPN training. Setup verified.\n")
        return None

    print("\n  Training CPN-EN (runs to --stop-pct or --max-epochs)...\n")
    loss_history = exp.run()
    max_recov = loss_history.max_pct_recov

    print("\n" + "-" * 72)
    print(f"  RESULT for {lesion_key} [{label}]: max recovery = {max_recov * 100:.1f}%")
    print("-" * 72 + "\n")
    sys.stdout.flush()
    return max_recov


def main():
    p = argparse.ArgumentParser(
        description="Stage 1: run CPN-EN training under a chosen emulator-error axis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--error", choices=["none", "parametric", "structural", "distributional"],
                   default="none", help="Which emulator-error axis to inject.")
    p.add_argument("--sigma-p", type=float, default=0.0,
                   help="Parametric magnitude (used when --error parametric).")
    p.add_argument("--en-num-neurons", type=int, default=None,
                   help="Structural: EN neuron count (repo default in_dim+50 = 87).")
    p.add_argument("--en-cell-type", choices=["lstm", "gru", "rnn"], default=None,
                   help="Structural: EN cell type (repo default lstm).")
    p.add_argument("--drift-delta", type=float, default=0.0,
                   help="Distributional magnitude (used when --error distributional; "
                        "sweep 0.0 .. 0.4).")
    p.add_argument("--drift-k", type=int, default=8,
                   help="Distributional: drift subspace dimensionality. Default 8.")
    p.add_argument("--drift-seed", type=int, default=42,
                   help="Distributional: drift direction seed. Default 42.")

    # --- Shared knobs, same as the stage0_repro ---
    p.add_argument("--lesion", choices=["f5m1", "aip", "m1", "all"], default="f5m1")
    p.add_argument("--device", default="cpu",
                   help="'cpu', or a CUDA device index like '0'. Default: cpu.")
    p.add_argument("--log-dir", default="./models")
    p.add_argument("--recover", action="store_true",
                   help="Use the pre-recovered (co-adaptation) mRNN. f5m1 only.")
    p.add_argument("--coadapt", nargs="?", const="on", default="off",
                   choices=["off", "on", "upstream"], help="Co-adaptation mode. Default off.")
    p.add_argument("--stim-sigma", type=float, default=2.175)
    p.add_argument("--en-ready-floor", default=None,
                   help="EN->CPN readiness floor (bare value or per-lesion pairs).")
    p.add_argument("--stop-pct", type=float, default=0.70,
                   help="Stop once max recovery reaches this fraction. Default 0.70.")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Hard epoch cap. Default: no cap.")
    p.add_argument("--baseline-only", action="store_true",
                   help="Build everything and print baselines, then stop before training.")
    args = p.parse_args()

    if args.error == "parametric" and args.sigma_p <= 0.0:
        raise SystemExit("--error parametric requires --sigma-p > 0.")
    if args.error == "distributional" and args.drift_delta <= 0.0:
        raise SystemExit("--error distributional requires --drift-delta > 0.")
    if args.error == "structural" and args.en_num_neurons is None and args.en_cell_type is None:
        raise SystemExit("--error structural requires --en-num-neurons and/or --en-cell-type.")

    # Error-injection runs are expected to hold true recovery below --stop-pct (that is what the error does). 
    # Without an epoch cap the run then never reaches the stop threshold and trains indefinitely. 
    # Give a warning and suggest an epoch cap.
    if args.error != "none" and args.max_epochs is None:
        print(
            "WARNING: --error " + args.error + " is injected but --max-epochs is "
            "not set. The injected error is expected to keep true recovery below "
            "--stop-pct (" + str(args.stop_pct) + "), so this run may never reach "
            "the stop threshold and will train indefinitely. Set --max-epochs to "
            "cap it and measure the (degraded) recovery, e.g. --max-epochs 2000.",
            flush=True,
        )

    device = resolve_device(args.device)
    os.makedirs(args.log_dir, exist_ok=True)
    floor_per_lesion, floor_fallback = parse_en_ready_floor(args.en_ready_floor)
    coadapt = resolve_coadapt(args.coadapt)

    cpn_epoch_cpn.STOP_PCT = args.stop_pct
    cpn_epoch_cpn.MAX_EPOCHS = args.max_epochs

    # Only the selected axis is active; others stay at repo defaults.
    sigma_p = args.sigma_p if args.error == "parametric" else 0.0
    en_num_neurons = args.en_num_neurons if args.error == "structural" else None
    en_cell_type = args.en_cell_type if args.error == "structural" else None
    drift_delta = args.drift_delta if args.error == "distributional" else 0.0

    is_all = args.lesion == "all"
    keys = ["f5m1", "aip", "m1"] if is_all else [args.lesion]

    summary = {}
    for key in keys:
        recover = (args.recover and LESIONS[key][1]) if is_all else args.recover
        floor = floor_per_lesion.get(key, floor_fallback)
        summary[key] = run_one(
            key, device, args.log_dir, recover, coadapt, args.stim_sigma,
            floor, args.baseline_only, args.error, sigma_p,
            en_num_neurons, en_cell_type, drift_delta, args.drift_k, args.drift_seed,
        )

    if not args.baseline_only:
        print("=" * 72, flush=True)
        print("STAGE 1 SUMMARY", flush=True)
        print(f"  error={describe_error(args.error, sigma_p, en_num_neurons, en_cell_type, drift_delta)}  "
              f"stim_sigma={args.stim_sigma}  stop_pct={args.stop_pct}  "
              f"max_epochs={args.max_epochs}", flush=True)
        for key in keys:
            r = summary[key]
            shown = "n/a" if r is None else f"{r * 100:.1f}%"
            print(f"  {key:5s}: max recovery = {shown}", flush=True)
        print("=" * 72, flush=True)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
