#!/usr/bin/env python3
"""
stage0_repro.py - Stage 0 reproduction harness for the Bryan et al. co-processor testbed.

Reproduces the vanilla-CPN baseline on the coproc-poc testbed and reports the
recovery percentage against the 70% Stage 0 gate: which requires the AIP, F5-M1,
and M1 lesions to all clear 70%.

Wraps the original repo's own config/staging/training/stats code: no re-implementation from scratch needed.

How to use it:
    # Fast setup check and baselines only (no training), on CPU:
    python stage0_repro.py --lesion f5m1 --device cpu --baseline-only

    # Full vanilla reproduction on GPU and checking the gate:
    python stage0_repro.py --lesion f5m1 --device 0
    python stage0_repro.py --lesion aip  --device 0

    # Run all three lesions, with a different EN readiness floor for each:
    python stage0_repro.py --lesion all --device 0 \
        --en-ready-floor f5m1=0.0006,aip=0.0012,m1=0.0012 --max-epochs 30000 --stop-pct 0.70

    # The co-adaptation comparison for AIP/M1 (upstream defaults coadapt=True here):
    python stage0_repro.py --lesion aip --device 0 --coadapt on --max-epochs 30000
    python stage0_repro.py --lesion aip --device 0 --coadapt upstream --max-epochs 30000
    python stage0_repro.py --lesion m1 --device 0 --coadapt on --max-epochs 30000
    python stage0_repro.py --lesion m1 --device 0 --coadapt upstream --max-epochs 30000

    # Actuation sweeps (all one-flag, no repo edits):
    python stage0_repro.py --lesion f5m1 --device 0 --stim-sigma 2.175 --num-stim-channels 32
    python stage0_repro.py --lesion f5m1 --device 0 --stim-sigma 2.175 --stim-decay 0.15

Key points:
    a) "Vanilla" baseline here means recover_after_lesion=False and coadapt=False (True in case of AIP and M1 lesions),
    matching the commented vanilla line in the upstream repo's train_cpn.ipynb. 
    This is baseline B1 in the proposal, and the healthy ceiling doubles as the oracle-bound B0.
    b) recover_after_lesion=True is only supported for the F5-M1 lesion out of the box, since that is the only pre-recovered mRNN model bundled with the upstreamrepo. 
    Use --recover to opt into it for f5m1; it is rejected for aip/m1.
    c) Training length is fixed inside the repo (ie:, in cpn_epoch_cpn.py: cpn_epoch_max_len is 100 for the M1 lesion, 200 otherwise). 
    To shorten a smoke test, lower that value in the repo; there is no run() argument for it.
    d) stim_sigma defaults to 2.175 in the repo AND in the Bryan et al. paper.

Co-adaptation defaults will adapt per lesion, as defined in the upstream repo:
    get_config              (f5m1) defaults coadapt=False
    get_aip_lesion_config   (aip)  defaults coadapt=True
    get_m1_lesion_config    (m1)   defaults coadapt=True

Stim/lesion windows are disjoint for aip and m1:
    aip: stim window is neurons 0-100 (num_stim_neurons defaults to 100, pads 0/200) but the lesion is at neurons 200-250.
    m1:  stim window is neurons 50-100 (num_stim_neurons=50, pads 50/200) but the lesion is at neurons 0-50.
    The CPN therefore has to act indirectly through recurrent connectivity for both.
    Expect much slower recovery for both these lesions compared to f5m1.

"""

import argparse
import logging
import os
import sys

# Surface the repo logs that progress through the root logger at INFO
_log = logging.getLogger()
_log.addHandler(logging.StreamHandler(sys.stdout))
_log.setLevel(logging.INFO)

import torch

import experiment            # noqa: E402  (repo package; import after logging setup)
import cpn                   # noqa: E402
import cpn_epoch_cpn         # noqa: E402  (for STOP_PCT / MAX_EPOCHS overrides)
import cpn_epoch_en          # noqa: E402  (for EN_READY_FLOOR override)


# Stage 0 recovery gate
GATE = 0.70

# Whatever the current repo defines, so an unset --en-ready-floor for one lesion in
# a multi-lesion sweep restores the repo value instead of leaking the previous
# lesion's override.
_REPO_EN_READY_FLOOR = getattr(cpn_epoch_en, "EN_READY_FLOOR", None)

# Human-readable lesion -> (builder name, whether recover_after_lesion is supported)
LESIONS = {
    "f5m1": ("default (connection) F5-M1 lesion", True),
    "aip":  ("AIP output lesion (neurons 200-250)", False),
    "m1":   ("M1 output lesion (neurons 0-50)", False),
}


def resolve_device(device_arg):
    """Return the cuda value the repo expects: None for CPU, or a device string like '0'."""
    if device_arg == "cpu":
        return None
    if not torch.cuda.is_available():
        raise SystemExit(
            f"Requested GPU device '{device_arg}' but torch.cuda.is_available() is False.\n"
            f"Your PyTorch is a CPU-only build. Reinstall a CUDA build, e.g.\n"
            f"  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            f"or run this harness with --device cpu."
        )
    return device_arg


def parse_en_ready_floor(spec):
    """Parse --en-ready-floor into (per_lesion_dict, fallback_value).

    Accepts either form, or a mix:
        0.0006                              -> fallback for every lesion
        aip=0.0012,m1=0.0012                -> per-lesion, others use the repo default
        0.0006,aip=0.0012                   -> 0.0006 fallback, aip overridden

    The emulator plateaus at a different pred_val_loss for each lesion, so one global floor tends to livelock some lesions 
    (gate never fires, every handoff falls through the 3000-epoch timeout) while starving others. Hence per-lesion values.
    """
    if spec is None:
        return {}, None

    per_lesion = {}
    fallback = None
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, _, raw = part.partition("=")
            key = key.strip()
            if key not in LESIONS:
                raise SystemExit(
                    f"--en-ready-floor: unknown lesion '{key}'. "
                    f"Valid keys: {', '.join(LESIONS)}."
                )
            try:
                per_lesion[key] = float(raw)
            except ValueError:
                raise SystemExit(f"--en-ready-floor: '{raw}' is not a number (for '{key}').")
        else:
            if fallback is not None:
                raise SystemExit(
                    "--en-ready-floor: give at most one bare fallback value; "
                    "use lesion=value for the rest."
                )
            try:
                fallback = float(part)
            except ValueError:
                raise SystemExit(f"--en-ready-floor: '{part}' is not a number.")

    return per_lesion, fallback


def resolve_coadapt(mode):
    """Map --coadapt mode to what build_config should pass.

    'off'      -> False everywhere (the clean vanilla B1 baseline)
    'on'       -> True everywhere
    'upstream' -> None, meaning: don't pass it, let each repo builder use its own
                  default (False for f5m1, True for aip and m1).
    """
    if mode == "upstream":
        return None
    return mode == "on"


def build_config(lesion_key, device, recover, coadapt, stim_sigma,
                 num_stim_channels, stim_decay, sigma_p = 0.0):
    """Construct the repo Config for one lesion, using the repo's own builders.

    num_stim_channels and stim_decay are forwarded via **kwargs into config.get,
    which every lesion builder shares, so they apply uniformly to f5m1/aip/m1.
    Passing None for any of coadapt/num_stim_channels/stim_decay means "use the
    repo default" (don't override).
    """
    common = dict(
        recover_after_lesion=recover,
        cuda=device,
        stim_sigma=stim_sigma,
        sigma_p=sigma_p,
    )
    # Only inject a knob when explicitly set, so an unspecified flag leaves the repo
    # default untouched rather than pinning it to our value. This matters most for
    # coadapt, where the aip/m1 builders default to True and f5m1 defaults to False.
    if coadapt is not None:
        common["coadapt"] = coadapt
    if num_stim_channels is not None:
        common["num_stim_channels"] = num_stim_channels
    if stim_decay is not None:
        common["stim_decay"] = stim_decay

    if lesion_key == "f5m1":
        # The default config path is the F5-M1 connection lesion.
        return experiment.experiment.get_config(**common)
    if lesion_key == "aip":
        return experiment.experiment.get_aip_lesion_config(**common)
    if lesion_key == "m1":
        return experiment.experiment.get_m1_lesion_config(**common)
    raise ValueError(f"Unknown lesion key: {lesion_key}")


def run_one(lesion_key, device, log_dir, recover, coadapt, stim_sigma,
            num_stim_channels, stim_decay, en_ready_floor, baseline_only):
    desc, recover_supported = LESIONS[lesion_key]
    if recover and not recover_supported:
        raise SystemExit(
            f"--recover is not available for the {lesion_key} lesion: no pre-recovered "
            f"mRNN model is bundled for it. Only the f5m1 lesion supports --recover."
        )

    # Set the EN readiness floor for THIS lesion before anything is built. With
    # --lesion all these run sequentially in one process, so this has to happen per
    # lesion, not once in main(). None restores whatever the repo/patch defines.
    cpn_epoch_en.EN_READY_FLOOR = (
        en_ready_floor if en_ready_floor is not None else _REPO_EN_READY_FLOOR
    )

    print("=" * 72)
    print(f"LESION: {lesion_key}  ->  {desc}")
    print(f"  device={'cpu' if device is None else 'cuda:' + str(device)}  "
          f"recover_after_lesion={recover}  "
          f"coadapt={'upstream default' if coadapt is None else coadapt}  "
          f"stim_sigma={stim_sigma}")
    print(f"  num_stim_channels={num_stim_channels if num_stim_channels is not None else 'repo default'}  "
          f"stim_decay={stim_decay if stim_decay is not None else 'repo default'}")
    print(f"  en_ready_floor={cpn_epoch_en.EN_READY_FLOOR}  "
          f"stop_pct={cpn_epoch_cpn.STOP_PCT}  max_epochs={cpn_epoch_cpn.MAX_EPOCHS}")
    print("=" * 72)

    cfg = build_config(lesion_key, device, recover, coadapt, stim_sigma,
                       num_stim_channels, stim_decay)

    coproc = cpn.CPN_EN_CoProc(cfg, log_dir=log_dir, recycle_thresh=1)
    exp = experiment.experiment.stage(coproc, cfg)

    # Baselines are computed during staging: read them straight off the experiment.
    healthy = exp.comp_loss_healthy.item()
    lesioned = exp.comp_loss_lesioned.item()
    healthy_hand = exp.comp_loss_healthy_hand.item()
    lesioned_hand = exp.comp_loss_lesioned_hand.item()

    print(f"\n  Healthy ceiling  (MSE, no lesion)        : {healthy:.6f}")
    print(f"  Lesioned floor   (MSE, lesion, no stim)    : {lesioned:.6f}")
    print(f"  Headroom (lesioned - healthy)              : {lesioned - healthy:.6f}")
    print(f"  Hand-muscle ceiling / floor                : {healthy_hand:.6f} / {lesioned_hand:.6f}")

    if lesioned <= healthy:
        print("\n  WARNING: lesioned loss is not worse than healthy loss. The lesion is not "
              "degrading behaviour as expected; recovery percentages will be meaningless. "
              "Check the lesion selection before proceeding.")

    if baseline_only:
        print("\n  --baseline-only set: skipping CPN training. Setup verified.\n")
        return None

    print("\n  Training vanilla CPN (this runs to --stop-pct or --max-epochs)...\n")
    loss_history = exp.run()

    # Best recovery achieved across all epochs, as defined in experiment/stats.py:
    # Recovery percentage: recov_pct = 1.0 - (task_loss - healthy_loss) / (lesioned_loss - healthy_loss)
    max_recov = loss_history.max_pct_recov
    passed = max_recov >= GATE

    print("\n" + "-" * 72)
    print(f"  RESULT for {lesion_key}: max recovery = {max_recov:.3f} "
          f"({max_recov * 100:.1f}%)   gate = {GATE:.0%}   "
          f"-> {'PASS' if passed else 'FAIL'}")
    print("-" * 72 + "\n")
    sys.stdout.flush()
    return max_recov


def main():
    p = argparse.ArgumentParser(
        description="Stage 0 reproduction harness for the coproc-poc co-processor testbed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--lesion", choices=["f5m1", "aip", "m1", "all"], default="f5m1",
                   help="Which lesion to reproduce. 'all' runs f5m1, aip, m1 in sequence.")
    p.add_argument("--device", default="cpu",
                   help="'cpu', or a CUDA device index like '0'. Default: cpu.")
    p.add_argument("--log-dir", default="./models",
                   help="Local directory for model checkpoints. Default: ./models ")
    p.add_argument("--recover", action="store_true",
                   help="Use the pre-recovered (co-adaptation) mRNN. f5m1 only.")

    # Bare --coadapt still means "on", so old invocations keep working.
    p.add_argument("--coadapt", nargs="?", const="on", default="off",
                   choices=["off", "on", "upstream"],
                   help="Co-adaptation mode. 'off' (default) forces coadapt=False for "
                        "every lesion -- the clean vanilla B1 baseline. 'on' forces True. "
                        "'upstream' passes nothing and lets each repo builder use its own "
                        "default, which is False for f5m1 but True for aip and m1. Bare "
                        "--coadapt means 'on'.")

    p.add_argument("--stim-sigma", type=float, default=2.175,
                   help="Stimulation Gaussian spread. Repo and paper default 2.175; set "
                        "to match the Methods section.")

    # --- Actuation-model knobs (pass straight through to config.get; no repo edit) ---
    p.add_argument("--num-stim-channels", type=int, default=None,
                   help="Number of stimulation channels (virtual electrodes). Repo "
                        "default is 16. More channels tile the neuron span more densely, "
                        "so a narrow stim_sigma still covers it -- the main lever for "
                        "reducing sigma sensitivity. NOTE: changes the physical actuation "
                        "model; report as a deliberate choice, not a reproduction.")
    p.add_argument("--stim-decay", type=float, default=None,
                   help="Stimulation temporal decay per step. Repo default is 0.3. A "
                        "smaller value lets stimulation persist and integrate over time, "
                        "which can compensate for narrower spatial spread. NOTE: changes "
                        "the temporal dynamics; report as a deliberate choice.")

    # --- Scheduling / stopping knobs (override repo module-level constants) ---
    p.add_argument("--en-ready-floor", default=None,
                   help="EN->CPN readiness floor (pred_val_loss threshold). Either a bare "
                        "value for every lesion (0.0006) or per-lesion pairs "
                        "(aip=0.0012,m1=0.0012), optionally with one bare fallback "
                        "(0.0006,aip=0.0012). Each lesion's emulator plateaus at a "
                        "different pred_val_loss, so a single global floor livelocks some "
                        "lesions and starves others. Too high livelocks (EN never "
                        "re-qualifies, handoff only via the 3000-epoch timeout); too low "
                        "starves the CPN. Unset lesions use the repo default.")
    p.add_argument("--stop-pct", type=float, default=0.70,
                   help="Stop once max recovery reaches this fraction. Default: the 0.70 "
                        "Stage 0 gate. Upstream stops only at 0.99.")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Hard epoch cap so a non-converging run still terminates and "
                        "reports its best recovery. Default: no cap (runs to --stop-pct).")

    p.add_argument("--baseline-only", action="store_true",
                   help="Build everything and print the healthy/lesioned baselines, then "
                        "stop before training. Fast way to verify the setup.")
    p.add_argument("--drift-delta", type=float, default=0.0,
               help="Distributional error drift magnitude (sweep 0.0 to 0.4).")
    p.add_argument("--sigma-p", type=float, default=0.0,
               help="Parametric error scale sigma_p for distorting EN weights.")
    args = p.parse_args()

    device = resolve_device(args.device)
    os.makedirs(args.log_dir, exist_ok=True)

    floor_per_lesion, floor_fallback = parse_en_ready_floor(args.en_ready_floor)
    coadapt = resolve_coadapt(args.coadapt)

    # Global stopping overrides; the per-lesion floor is applied inside run_one.
    cpn_epoch_cpn.STOP_PCT = args.stop_pct
    cpn_epoch_cpn.MAX_EPOCHS = args.max_epochs

    is_all = args.lesion == "all"
    keys = ["f5m1", "aip", "m1"] if is_all else [args.lesion]

    summary = {}
    for key in keys:
        # For just one explicit lesion, pass --recover through so run_one can reject it if unsupported.
        # For 'all', silently skip --recover on the unsupported lesions instead of aborting the full sweep.
        recover = (args.recover and LESIONS[key][1]) if is_all else args.recover
        floor = floor_per_lesion.get(key, floor_fallback)
        result = run_one(
            key, device, args.log_dir, recover, coadapt,
            args.stim_sigma, args.num_stim_channels, args.stim_decay,
            floor, args.baseline_only,
        )
        summary[key] = result

    if not args.baseline_only:
        print("=" * 72, flush=True)
        print("STAGE 0 SUMMARY", flush=True)
        print(f"  coadapt={args.coadapt}  stim_sigma={args.stim_sigma}  "
              f"stop_pct={args.stop_pct}  max_epochs={args.max_epochs}", flush=True)
        for key in keys:
            r = summary[key]
            verdict = "PASS" if (r is not None and r >= GATE) else "FAIL"
            shown = "n/a" if r is None else f"{r * 100:.1f}%"
            floor = floor_per_lesion.get(key, floor_fallback)
            floor_shown = "repo default" if floor is None else f"{floor:g}"
            print(f"  {key:5s}: {shown:>7s}  -> {verdict}   (en_ready_floor={floor_shown})",
                  flush=True)

        # The full Stage 0 gate needs all three lesions; only assert it when all ran.
        required = ("f5m1", "aip", "m1")
        have_all = all(k in summary for k in required)
        if have_all:
            gate_ok = all(
                summary[k] is not None and summary[k] >= GATE for k in required
            )
            print(f"\n  Gate (AIP, F5-M1, and M1 all >= {GATE:.0%}): "
                  f"{'PASSED - Stage 0 complete!' if gate_ok else 'FAILED'}", flush=True)
        else:
            ran = ", ".join(keys)
            print(f"\n  Gate: not evaluated (needs f5m1 + aip + m1; this run did only {ran}).",
                  flush=True)
        print("=" * 72, flush=True)
        sys.stdout.flush()


if __name__ == "__main__":
    main()