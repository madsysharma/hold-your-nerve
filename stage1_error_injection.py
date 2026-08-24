#!/usr/bin/env python3
"""
stage1_error_injection.py - run CPN-EN training with a chosen emulator-error axis.

Summary:
Where Stage 0 reproduces the clean testbed, Stage 1 injects a controlled emulator error and runs the same CPN-EN training so one
can see how recovery degrades. Mirrors stage0_repro's CLI and reuses its helpers.

The three benchmark axes and how they enter training:

  parametric     : cfg.sigma_p -> ErrorInjector.apply_param_error on the EN at the
                   EN->CPN handoff (cpn.py).
  structural     : cfg.en_num_neurons / cfg.en_cell_type -> ErrorInjector.build_structural_en
                   builds the EN with a mismatched capacity or cell type (cpn_epoch_en.py).
  distributional : cfg.drift_delta -> ErrorInjector.apply_dist_error installs a forward
                   hook on the brain (self.mike) at the EN->CPN handoff (experiment.py),
                   drifting the brain state x->Rx+b while the EN is frozen stale (cpn.py).

Examples:
  python stage1_error_injection.py --error parametric --sigma-p 0.2 --lesion f5m1
  python stage1_error_injection.py --error structural --en-num-neurons 40 --lesion m1
  python stage1_error_injection.py --error structural --en-cell-type gru --lesion aip
  python stage1_error_injection.py --error distributional --drift-delta 0.2 --lesion f5m1
  python stage1_error_injection.py --error none --baseline-only
"""
import argparse
import json
import os
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

import warnings
warnings.filterwarnings("ignore")


def build_config_with_error(lesion_key, device, recover, coadapt, stim_sigma,
                            warm_start_en=None, pretrain_en_out=None, uncertainty_estimator="disagreement",
                            reliability_log=None, conformal=False,
                            conformal_threshold=1.0, conformal_safe_norm=2.0,
                            sigma_p=0.0, en_num_neurons=None, en_cell_type=None,
                            drift_delta=0.0, drift_k=8, drift_seed=42, drift_ramp=-1,
                            ta_lambda=0.0, ta_beta=0.0, ta_caution="soft",
                            ta_halt_threshold=1.0, ta_halt_penalty=10.0, ta_halt_temperature=None,
            num_members=1, dr_sigma_max=0.0, checkpoint_en=False, oracle=False):
    """Like stage0_repro.build_config, but also threads the error-injection knobs
    (parametric sigma_p; structural en_num_neurons/en_cell_type; distributional
    drift_delta/drift_k/drift_seed) into the repo Config. 
    Unset knobs are omitted so the repo defaults stand.
    """
    common = dict(
        recover_after_lesion=recover,
        cuda=device,
        stim_sigma=stim_sigma,
        sigma_p=sigma_p,
        drift_delta=drift_delta,
        drift_k=drift_k,
        drift_seed=drift_seed,
        drift_ramp=drift_ramp,
        ta_lambda=ta_lambda, ta_beta=ta_beta, ta_caution=ta_caution,
        ta_halt_threshold=ta_halt_threshold, ta_halt_penalty=ta_halt_penalty,
        ta_halt_temperature=ta_halt_temperature,
    )
    if warm_start_en is not None:
        common["warm_start_en_path"] = warm_start_en
    if pretrain_en_out is not None:
        common["pretrain_en_out"] = pretrain_en_out
    common["uncertainty_estimator"] = uncertainty_estimator
    if reliability_log is not None:
        common["reliability_log_path"] = reliability_log
    if conformal:
        common["conformal_enabled"] = True
        common["conformal_width_threshold"] = conformal_threshold
        common["conformal_safe_norm"] = conformal_safe_norm
    if oracle:
        common["oracle"] = True
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
            en_num_neurons, en_cell_type, drift_delta, drift_k, drift_seed,
            drift_ramp=-1, seed=0, ta_lambda=0.0, ta_beta=0.0, ta_caution="soft",
            ta_halt_threshold=1.0, ta_halt_penalty=10.0, ta_halt_temperature=None,
            num_members=1, dr_sigma_max=0.0,
            checkpoint_en=False, warm_start_en=None, pretrain_en_out=None,
            uncertainty_estimator="disagreement", reliability_log=None,
            conformal=False, conformal_threshold=1.0, conformal_safe_norm=2.0,
            oracle=False):
    torch.manual_seed(seed)   # reproducible init per (config, seed)
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
        drift_ramp=drift_ramp, ta_lambda=ta_lambda, ta_beta=ta_beta,
        ta_caution=ta_caution, ta_halt_threshold=ta_halt_threshold,
        ta_halt_penalty=ta_halt_penalty, ta_halt_temperature=ta_halt_temperature,
        warm_start_en=warm_start_en, pretrain_en_out=pretrain_en_out,
        uncertainty_estimator=uncertainty_estimator,
        reliability_log=reliability_log,
        conformal=conformal, conformal_threshold=conformal_threshold,
        conformal_safe_norm=conformal_safe_norm,
        oracle=oracle,
    )

    if oracle:
        # B0: EN replaced by the true mRNN (upper bound, Delta=0 by construction).
        from oracle import OracleCoProc
        coproc = OracleCoProc(cfg, log_dir=log_dir, recycle_thresh=1)
    elif dr_sigma_max and dr_sigma_max > 0.0:
        # Domain-randomized emulator (B2): CPN trains against a range of
        # per-trial parametric perturbations of the EN.
        from domain_random import DomainRandomizedCoProc
        coproc = DomainRandomizedCoProc(cfg, sigma_max=dr_sigma_max, seed=seed,
                                        log_dir=log_dir, recycle_thresh=1)
    elif num_members and num_members > 1:
        # Ensemble emulator (B4/B5/B6): M members wired as the EN so the
        # trust-aware lambda term has ensemble disagreement to read.
        from stage2_ensemble_val import WiredCPN_EN_CoProc
        coproc = WiredCPN_EN_CoProc(cfg, num_members=num_members, base_seed=seed,
                                    log_dir=log_dir, recycle_thresh=1,
                                    checkpoint=checkpoint_en)
    else:
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

    # Dual-recovery + safety readout at the peak-true-recovery operating point.
    eta_true = loss_history.true_at_max_recov
    eta_hat = loss_history.apparent_at_max_recov
    delta = loss_history.gap_at_max_recov
    cvar_val = loss_history.cvar_at_max_recov
    dmg = loss_history.damage_frac_at_max_recov
    eta_hat_s = f"{eta_hat * 100:.1f}%" if eta_hat is not None else "n/a"
    delta_s = f"{delta * 100:+.1f}pp" if delta is not None else "n/a"
    cvar_s = f"{cvar_val:.4e}" if cvar_val is not None else "n/a"
    dmg_s = f"{dmg * 100:.1f}%" if dmg is not None else "n/a"
    print("\n" + "-" * 72)
    print(f"  RESULT for {lesion_key} [{label}]:")
    eta_true_s = f"{eta_true * 100:.1f}%" if eta_true is not None else f"{max_recov * 100:.1f}%"
    print(f"    true recovery   eta      = {eta_true_s} (at CPN operating point)")
    print(f"    apparent recov  eta_hat  = {eta_hat_s}   (what the CPN sees via the EN)")
    print(f"    exploitation gap Delta   = {delta_s}   (eta_hat - eta)")
    print(f"    stim-cost CVaR (0.95)    = {cvar_s}")
    print(f"    damage-proxy fraction    = {dmg_s}   (Shannon k; violation metric)")
    print("-" * 72 + "\n")
    sys.stdout.flush()
    return {
        "eta_true": eta_true if eta_true is not None else max_recov,
        "eta_hat": eta_hat,
        "delta": delta,
        "cvar": cvar_val,
        "damage_frac": dmg,
        "max_recov": max_recov,
    }


def main():
    p = argparse.ArgumentParser(
        description="Stage 1: run CPN-EN training under a chosen error axis.",
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
    p.add_argument("--oracle", action="store_true",
                   help="B0: replace the EN with the true mRNN (upper bound; Delta=0 by "
                        "construction). Skips EN pre-training (the oracle is already perfect).")
    p.add_argument("--conformal", action="store_true",
                   help="B6: enable the ACI conformal amplitude cap (task 4.2) on the "
                        "co-processor stim.")
    p.add_argument("--conformal-threshold", type=float, default=1.0,
                   help="Cap when the conformal interval exceeds this width (calibrate ~ c*W_0).")
    p.add_argument("--conformal-safe-norm", type=float, default=2.0,
                   help="Capped stim L2 norm when the interval is wide (~ amp-at-k damage limit).")
    p.add_argument("--reliability-log", default=None,
                   help="Write (ensemble uncertainty, emulator error) pairs at eval to this "
                        "path, for the H4 reliability diagram (analysis.py --do reliability).")
    p.add_argument("--uncertainty-estimator", default="disagreement",
                   choices=["disagreement", "max_frobenius", "single_var", "mc_dropout"],
                   help="Ensemble uncertainty estimator for the trust-aware lambda term "
                        "(B4/B5). max_frobenius/single_var add trained variance heads; "
                        "mc_dropout adds dropout + stochastic passes.")
    p.add_argument("--pretrain-en-out", default=None,
                   help="EN-only mode: train the EN to convergence, save it to this path, "
                        "then stop before any CPN training. Run once per lesion with "
                        "--error none and a SINGLE EN (no --num-members / --dr-sigma-max) "
                        "to mint clean warm-start ENs.")
    p.add_argument("--warm-start-en", default=None,
                   help="Path to a converged en.model (or a run-log dir containing one) "
                        "to load INSTEAD of training the EN from scratch -- skips most "
                        "pre-training. Must match this run's lesion + EN architecture.")
    p.add_argument("--checkpoint-en", action="store_true",
                   help="Gradient-checkpoint the EN rollout (recompute activations in "
                        "backward). Cuts memory for ensemble baselines (B4/B5) at some "
                        "compute cost; results are identical.")
    p.add_argument("--dr-sigma-max", type=float, default=0.0,
                   help="Domain randomization (B2): per-trial sigma_p ~ U(0, this). >0 wraps "
                        "the EN so the CPN trains against a range of perturbations. 0 = off.")
    p.add_argument("--num-members", type=int, default=1,
                   help="Ensemble size M for the emulator. >1 wires M members as the "
                        "EN (needed for the trust-aware lambda term, B4/B5/B6). Default 1 = single EN.")
    p.add_argument("--ta-lambda", type=float, default=0.0,
                   help="Trust-aware uncertainty weight (needs an ensemble EN). Default 0 = off.")
    p.add_argument("--ta-beta", type=float, default=0.0,
                   help="Trust-aware safety-cost weight (stim energy). Default 0 = off.")
    p.add_argument("--ta-caution", choices=["soft", "hard"], default="soft",
                   help="soft = MOPO graded penalty; hard = MOReL HALT.")
    p.add_argument("--ta-halt-threshold", type=float, default=1.0)
    p.add_argument("--ta-halt-temperature", type=float, default=None)
    p.add_argument("--ta-halt-penalty", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0,
                   help="Manual seed for reproducible EN/CPN init.")
    p.add_argument("--metrics-json", default=None,
                   help="If set, write the per-lesion metrics dict to this JSON path.")
    p.add_argument("--drift-ramp", type=int, default=-1,
                   help="Distributional timescale: -1 = slow ramp within one trial, restarting each trial (default); 0 = instantaneous onset; N > 0 = slow ramp ACROSS trials over N total steps (multi-session MINDFUL drift that accumulates).")

    # --- Shared knobs, same meaning as stage0_repro ---
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

    # Error-injection runs are expected to hold true recovery below --stop-pct.
    # Without an epoch cap the run then never reaches the stop threshold and trains indefinitely. Warn the user and suggest a cap.
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

    # Only the selected axis is active while others stay at repo defaults.
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
            args.drift_ramp, seed=args.seed, ta_lambda=args.ta_lambda,
            ta_beta=args.ta_beta, ta_caution=args.ta_caution, ta_halt_temperature = args.ta_halt_temperature,
            ta_halt_threshold=args.ta_halt_threshold, ta_halt_penalty=args.ta_halt_penalty,
            num_members=args.num_members, dr_sigma_max=args.dr_sigma_max,
            checkpoint_en=args.checkpoint_en, warm_start_en=args.warm_start_en,
            pretrain_en_out=args.pretrain_en_out,
            uncertainty_estimator=args.uncertainty_estimator,
            reliability_log=args.reliability_log,
            conformal=args.conformal, conformal_threshold=args.conformal_threshold,
            conformal_safe_norm=args.conformal_safe_norm, oracle=args.oracle,
        )

    if not args.baseline_only:
        print("=" * 72, flush=True)
        print("STAGE 1 SUMMARY", flush=True)
        print(f"  error={describe_error(args.error, sigma_p, en_num_neurons, en_cell_type, drift_delta)}  "
              f"stim_sigma={args.stim_sigma}  stop_pct={args.stop_pct}  "
              f"max_epochs={args.max_epochs}", flush=True)
        for key in keys:
            r = summary[key]
            shown = "n/a" if r is None else f"{r['eta_true'] * 100:.1f}%"
            print(f"  {key:5s}: true recovery = {shown}", flush=True)
        print("=" * 72, flush=True)

    if args.metrics_json:
        with open(args.metrics_json, "w") as _f:
            json.dump(summary, _f, indent=2)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
