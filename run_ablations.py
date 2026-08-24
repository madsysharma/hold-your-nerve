#!/usr/bin/env python3
"""
run_ablations.py -- one-knob-at-a-time ablations for the trust-aware method.

This is the "ablation matrix" from the research doc. It is deliberately SEPARATE
from run_experiments.py: the main sweep is a factorial (baselines x axes x
severities x lesions x seeds), whereas an ablation varies exactly ONE component
from a fixed reference config. Folding ablations into the main grid would explode
it combinatorially with cells you would never actually run.

It reuses run_experiments' execution machinery (execute_single_run, build_command,
the metrics parsing) so every ablation record has the SAME on-disk format as a main
run -- analysis.py then reads both the same way.

Ablation axes (research doc):
  lambda     caution weight in {0, 0.1, 1, 10, 100}                 [runnable]
  beta       safety weight toggled {0, 0.5}                          [runnable]
  members    ensemble size M in {2, 3, 5, 10}                        [runnable]
  caution    hard vs soft pessimism (B5 vs B4)                       [runnable]
  conformal  conformal layer off vs on (B4 vs B6)                    [SKIPPED: B6 not wired]
  estimator  disagreement / max_frobenius / mc_dropout / single_var  [runnable]

The last two emit honest "skipped" records for the parts that depend on unbuilt
prerequisites (the conformal cap in the CPN forward; member variance heads +
dropout modules + a --uncertainty-estimator flag). They are never fabricated.

Reference config: the ablation holds everything else fixed at a point where the
method actually matters -- by default B4 (soft trust-aware), on a moderate
parametric error condition (that is where an emulator error exists to exploit),
with the clean anchor included for contrast. Override with the flags below.

Example:
  python run_ablations.py --max-epochs 500000 --devices 0 \
      --ablations lambda beta members caution --lesion AIP --seeds 42 101
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment"))
import run_experiments as R   # reuse the runner's machinery


# ---- which ablation axes exist, and how each maps to a stage1 override ----

LAMBDA_VALUES   = [0.0, 0.1, 1.0, 10.0, 100.0]
BETA_VALUES     = [0.0, 0.5]
MEMBER_VALUES   = [2, 3, 5, 10]
CAUTION_VALUES  = ["soft", "hard"]
ESTIMATORS      = ["disagreement", "max_frobenius", "mc_dropout", "single_var"]
WIRED_ESTIMATORS = {"disagreement"}   # the only one usable in training today

ABLATION_AXES = ["lambda", "beta", "members", "caution", "conformal", "estimator"]


def _base_config(run_id, lesion, axis, sev, baseline, seed, extra_flags=None,
                 ta_lambda=None, ablation=None, ablation_value=None):
    cfg = {
        "run_id": run_id,
        "lesion": lesion, "error_axis": axis, "severity": sev,
        "baseline": baseline, "seed": seed, "status": "pending",
        "ablation": ablation, "ablation_value": ablation_value,
    }
    if extra_flags:
        cfg["extra_flags"] = extra_flags
    if ta_lambda is not None:
        cfg["ta_lambda"] = ta_lambda
    return cfg


def build_ablation_configs(axes, conditions, baseline, seeds):
    """One config list, tagged by which knob is varied. Everything not being
    ablated stays at the reference (baseline's default flags)."""
    configs = []
    n = 0

    def add(cfg):
        nonlocal n
        n += 1
        configs.append(cfg)

    for (axis_name, sev) in conditions:
        for seed in seeds:
            tag = f"{axis_name}.{sev}.s{seed}"

            if "lambda" in axes:
                for v in LAMBDA_VALUES:
                    add(_base_config(f"abl_lambda_{v}_{tag}", None, axis_name, sev,
                                     baseline, seed, ta_lambda=v,
                                     ablation="lambda", ablation_value=v))

            if "beta" in axes:
                for v in BETA_VALUES:
                    add(_base_config(f"abl_beta_{v}_{tag}", None, axis_name, sev,
                                     baseline, seed, extra_flags=["--ta-beta", v],
                                     ablation="beta", ablation_value=v))

            if "members" in axes:
                for v in MEMBER_VALUES:
                    add(_base_config(f"abl_M_{v}_{tag}", None, axis_name, sev,
                                     baseline, seed, extra_flags=["--num-members", v],
                                     ablation="members", ablation_value=v))

            if "caution" in axes:
                for v in CAUTION_VALUES:
                    # soft == B4, hard == B5; both are runnable baselines
                    base = "B4" if v == "soft" else "B5"
                    add(_base_config(f"abl_caution_{v}_{tag}", None, axis_name, sev,
                                     base, seed,
                                     ablation="caution", ablation_value=v))

            if "conformal" in axes:
                # off == B4 (no cap); on == B6 (cap). B6 is not wired, so its config
                # will be auto-skipped by execute_single_run (not fabricated).
                add(_base_config(f"abl_conformal_off_{tag}", None, axis_name, sev,
                                 "B4", seed, ablation="conformal", ablation_value="off"))
                add(_base_config(f"abl_conformal_on_{tag}", None, axis_name, sev,
                                 "B6", seed, ablation="conformal", ablation_value="on"))

            if "estimator" in axes:
                for est in ESTIMATORS:
                    # All four are wired now: disagreement (default), max_frobenius /
                    # single_var (trained variance heads), mc_dropout (dropout passes).
                    add(_base_config(f"abl_est_{est}_{tag}", None, axis_name, sev,
                                     baseline, seed,
                                     extra_flags=["--uncertainty-estimator", est],
                                     ablation="estimator", ablation_value=est))

    # assign the lesion (single lesion per invocation keeps warm-start EN valid)
    return configs


def main():
    p = argparse.ArgumentParser(description="One-knob ablations for the trust-aware method.")
    p.add_argument("--max-epochs", type=int, required=True)
    p.add_argument("--ablations", nargs="+", default=["lambda", "beta", "members", "caution"],
                   choices=ABLATION_AXES,
                   help="Which ablation axes to run. conformal/estimator include honest skips "
                        "for the parts that need unbuilt prerequisites.")
    p.add_argument("--baseline", default="B4",
                   help="Reference baseline held fixed while one knob varies (default B4, soft).")
    p.add_argument("--lesion", default="AIP", help="Single lesion (AIP / M1 / F5_M1).")
    p.add_argument("--axes", nargs="+", default=["none", "parametric"],
                   help="Error axes for the reference conditions.")
    p.add_argument("--severities", nargs="+", default=None,
                   help="Severities to include; default = clean anchor + the middle "
                        "level of each chosen axis (where the method actually matters).")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 2024])
    p.add_argument("--results-dir", default="ablation_results")
    p.add_argument("--stim-sigma", type=float, default=2.175)
    p.add_argument("--en-ready-floor", type=int, default=None)
    p.add_argument("--warm-start-en", default=None)
    p.add_argument("--coadapt", nargs="?", const="on", default="off")
    p.add_argument("--log-reliability", action="store_true")
    p.add_argument("--devices", nargs="+", default=["cpu"])
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    opts = p.parse_args()

    # single device: build_command reads opts.devices[0] via _current_device.
    R._WORKER_DEVICE = None

    # build the reference conditions: clean anchor + a representative level per axis
    conditions = []
    for axis in opts.axes:
        levels = R.AXIS_LEVELS.get(axis, [])
        if not levels:
            continue
        if axis == "none":
            conditions.append(("none", levels[0][0]))
            continue
        if opts.severities:
            for sev, _ in levels:
                if sev in opts.severities:
                    conditions.append((axis, sev))
        else:
            conditions.append((axis, levels[len(levels) // 2][0]))   # middle severity
    if not conditions:
        print("No valid (axis, severity) conditions; check --axes/--severities.")
        return

    configs = build_ablation_configs(opts.ablations, conditions, opts.baseline, opts.seeds)
    for c in configs:
        c["lesion"] = opts.lesion
    if opts.limit:
        configs = configs[:opts.limit]

    os.makedirs(opts.results_dir, exist_ok=True)
    with open(os.path.join(opts.results_dir, "ablation_configs.json"), "w") as f:
        json.dump(configs, f, indent=2)

    print("=" * 64)
    print(f"ABLATIONS: {opts.ablations}  baseline={opts.baseline}  lesion={opts.lesion}")
    print(f"conditions={conditions}  seeds={opts.seeds}  -> {len(configs)} runs")
    print("=" * 64)

    done = 0
    for cfg in configs:
        # pre-skipped (unwired estimator) records are written straight through
        if cfg.get("status") == "skipped":
            R._write(os.path.join(opts.results_dir, f"{cfg['run_id']}.json"), cfg)
        else:
            R.execute_single_run(cfg, opts)
        done += 1
        if done % 10 == 0:
            print(f"[{done}/{len(configs)}]")
    print(f"Done: {done} ablation runs -> {opts.results_dir}/")


if __name__ == "__main__":
    main()
