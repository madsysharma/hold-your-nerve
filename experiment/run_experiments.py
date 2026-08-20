"""
run_experiments.py -- factorial B-sweep runner.

This ACTUALLY runs the training pipeline per config and records the real metrics
that stage1_error_injection.py measures (true recovery eta, apparent recovery
eta_hat, exploitation gap Delta, stim-cost CVaR, damage-proxy fraction). It does
NOT fabricate results.

Severity is per-axis, not a shared magnitude. Parametric and distributional have a
genuine continuous magnitude (sigma_p, delta); structural does not -- its severity
is discrete (capacity, topology) -- so each axis carries its OWN named severity
ladder rather than borrowing one number across all three. "clean" (no error) is its
own `none` axis, run once per (lesion, baseline, seed) and shared as the anchor
point for every axis's trend, instead of being repeated inside each axis.

Each runnable config maps to a real stage1_error_injection.py invocation; the child
writes its metrics to a per-run JSON via --metrics-json, which this script reads
back. Config enumeration, the parallel pool, and per-run checkpoint/resume are
preserved.

Compute reality: --max-epochs only caps CPN epochs; the EN PRE-TRAINING that
precedes them is unbounded (it runs to the readiness floor), so every run pays the
full pre-training first -- hundreds of thousands of epochs (see the Stage 0 notes).
Set --timeout generously (hours) and use a pre-trained-EN warm-start, or raise
--en-ready-floor to reach CPN sooner. Use the subset flags to run tractable slices.
"""
import argparse
import functools
import itertools
import json
import multiprocessing as mp
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE1 = os.path.join(REPO_ROOT, "stage1_error_injection.py")

# Baseline -> extra stage1 flags. B1 (vanilla) and B3 (safety-penalty-only, beta>0
# lambda=0) run for real now: neither needs an ensemble. B4/B5/B6 use the lambda
# uncertainty term, which needs the ensemble wired as the in-loop EN (Stage 2 of
# the trust-aware integration) -- until then they are skipped, not faked. B0 (oracle)
# needs an oracle mode. Move a baseline into BASELINE_FLAGS to make it runnable.
BASELINE_FLAGS = {
    "B1": [],
    "B2": ["--dr-sigma-max", "0.2"],
    "B3": ["--ta-beta", "0.5"],
    "B4": ["--ta-lambda", "0.3", "--ta-caution", "soft", "--num-members", "5", "--checkpoint-en"],
    "B5": ["--ta-lambda", "0.3", "--ta-caution", "hard", "--num-members", "5", "--checkpoint-en"],
    # "B6": B4 + the ACI conformal cap in the CPN forward (conformal.py). Still
    # to wire; skipped until then.
}
RUNNABLE_BASELINES = set(BASELINE_FLAGS)

LESION_MAP = {"AIP": "aip", "M1": "m1", "F5_M1": "f5m1"}

# Device assigned to THIS worker process. In parallel runs the pool initializer
# hands each worker one device from --devices, round-robin; in serial runs main
# sets it directly. stage1 --device wants "cpu" or a GPU index string ("0").
_WORKER_DEVICE = None


def _current_device(opts):
    return _WORKER_DEVICE if _WORKER_DEVICE is not None else opts.devices[0]


def _init_worker(devices, counter):
    """Pool initializer: give each worker process a distinct device round-robin,
    so N workers on M GPUs actually spread across them instead of piling on one."""
    global _WORKER_DEVICE
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    _WORKER_DEVICE = devices[idx % len(devices)]
    print(f"[worker {idx}] device={_WORKER_DEVICE}", flush=True)

# Per-axis severity ladders: (severity_name, stage1 error flags). Each axis owns its
# own levels -- nothing borrows sigma_p's number. Structural levels come straight
# from the error definition: capacity 87 -> {20, 40, 160}, topology LSTM -> {GRU,
# vanilla RNN}. They are DISTINCT conditions, not points on one severity scalar;
# order them in analysis by measured effect rather than assuming monotonicity
# (over-capacity 160 in particular need not be "milder" than under-capacity 20).
AXIS_LEVELS = {
    "none": [
        ("clean", ["--error", "none"]),
    ],
    "parametric": [
        (f"sigma_p_{m}", ["--error", "parametric", "--sigma-p", str(m)])
        for m in (0.05, 0.1, 0.2, 0.4)
    ],
    "distributional": [
        (f"delta_{m}", ["--error", "distributional", "--drift-delta", str(m)])
        for m in (0.05, 0.1, 0.2, 0.4)
    ],
    "structural": [
        ("cap_20",   ["--error", "structural", "--en-num-neurons", "20"]),
        ("cap_40",   ["--error", "structural", "--en-num-neurons", "40"]),
        ("cap_160",  ["--error", "structural", "--en-num-neurons", "160"]),
        ("topo_gru", ["--error", "structural", "--en-cell-type", "gru"]),
        ("topo_rnn", ["--error", "structural", "--en-cell-type", "rnn"]),
    ],
}


def flags_for(axis, severity):
    """The stage1 error flags for a given (axis, severity), or None if unknown."""
    for sev, flags in AXIS_LEVELS.get(axis, []):
        if sev == severity:
            return flags
    return None


def generate_configs(output_path, lesions, axes, baselines, seeds):
    conditions = [(axis, sev) for axis in axes for sev, _ in AXIS_LEVELS.get(axis, [])]
    combos = itertools.product(lesions, conditions, baselines, seeds)
    configs = []
    for idx, (lesion, (axis, sev), base, seed) in enumerate(combos):
        configs.append({
            "run_id": f"run_{idx:04d}",
            "lesion": lesion, "error_axis": axis, "severity": sev,
            "baseline": base, "seed": seed, "status": "pending",
        })
    with open(output_path, "w") as f:
        json.dump(configs, f, indent=2)
    print(f"Generated {len(configs)} configurations in '{output_path}'.")
    return configs


def build_command(config, opts, metrics_path, log_dir):
    """Real stage1 invocation for a runnable config, or None if not runnable."""
    if config["baseline"] not in RUNNABLE_BASELINES:
        return None
    lesion = LESION_MAP.get(config["lesion"])
    if lesion is None:
        return None
    ef = flags_for(config["error_axis"], config["severity"])
    if ef is None:
        return None
    cmd = [sys.executable, STAGE1, "--lesion", lesion, *ef,
           *BASELINE_FLAGS.get(config["baseline"], []),
           "--device", _current_device(opts),
           "--seed", str(config["seed"]),
           "--max-epochs", str(opts.max_epochs),
           "--stim-sigma", str(opts.stim_sigma),
           "--log-dir", log_dir,
           "--metrics-json", metrics_path]
    if opts.en_ready_floor is not None:
        cmd += ["--en-ready-floor", str(opts.en_ready_floor)]
    return cmd


def execute_single_run(config, opts):
    checkpoint = os.path.join(opts.results_dir, f"{config['run_id']}.json")
    if os.path.exists(checkpoint):
        with open(checkpoint) as f:
            return json.load(f)

    base = config["baseline"]
    result = {**config}

    if base not in RUNNABLE_BASELINES:
        result["status"] = "skipped"
        result["reason"] = (f"{base} not yet runnable: B0 needs an oracle mode; "
                            f"B6 needs the ACI conformal cap wired into the CPN "
                            f"forward. Not fabricating a result.")
        _write(checkpoint, result)
        return result

    run_log = os.path.join(opts.results_dir, f"{config['run_id']}_logs")
    metrics_path = os.path.join(opts.results_dir, f"{config['run_id']}_metrics.json")
    os.makedirs(run_log, exist_ok=True)
    cmd = build_command(config, opts, metrics_path, run_log)
    if cmd is None:
        result["status"] = "skipped"
        result["reason"] = "no (axis, severity) -> flags mapping for this config"
        _write(checkpoint, result)
        return result

    if opts.dry_run:
        result["status"] = "dry_run"
        result["command"] = " ".join(cmd)
        return result

    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=opts.timeout)
    except subprocess.TimeoutExpired:
        result["status"] = "failed"
        result["error"] = f"timed out after {opts.timeout}s"
        _write(checkpoint, result)
        return result

    if proc.returncode != 0 or not os.path.exists(metrics_path):
        result["status"] = "failed"
        result["error"] = (proc.stderr or proc.stdout or "")[-800:]
        _write(checkpoint, result)
        return result

    with open(metrics_path) as f:
        per_lesion = json.load(f)
    m = per_lesion.get(LESION_MAP[config["lesion"]])   # real metrics for this lesion
    result["metrics"] = m                              # may be null if run never reached CPN epochs
    result["status"] = "completed" if m is not None else "completed_no_cpn"
    _write(checkpoint, result)
    return result


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", default="configs.json")
    p.add_argument("--results-dir", default="results")
    # Real-run parameters (passed through to stage1).
    p.add_argument("--max-epochs", type=int, required=True,
                   help="Per-run CPN-epoch cap. Does NOT bound EN pre-training, so a "
                        "run still pays the full pre-training before any CPN epoch.")
    p.add_argument("--stim-sigma", type=float, default=2.175)
    p.add_argument("--en-ready-floor", default=None,
                   help="Passed to stage1; raise it to reach CPN training sooner.")
    p.add_argument("--timeout", type=int, default=None,
                   help="Per-run wall-clock timeout in seconds (default: none). Set "
                        "this to HOURS -- EN pre-training is long.")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel runs. Default 1: training is heavy, so oversubscribing "
                        "CPUs/GPU thrashes. On a multi-GPU box set this to the GPU count.")
    p.add_argument("--devices", nargs="+", default=["cpu"],
                   help="Device(s) for the runs: 'cpu' or GPU index strings, e.g. "
                        "--devices 0 1 2 3. Assigned to workers round-robin. A single "
                        "device is used for every run.")
    # Subset controls, so you can run a tractable slice.
    p.add_argument("--lesions", nargs="+", default=["AIP", "M1", "F5_M1"])
    p.add_argument("--axes", nargs="+", default=list(AXIS_LEVELS.keys()),
                   help="Which error axes to include (default: none parametric "
                        "distributional structural). 'none' is the clean anchor.")
    p.add_argument("--severities", nargs="+", default=None,
                   help="Optional: restrict to these severity names (e.g. cap_20 "
                        "sigma_p_0.2). Default: all levels of the selected axes.")
    p.add_argument("--baselines", nargs="+", default=["B0", "B1", "B2", "B3", "B4", "B5", "B6"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 2024, 888, 999])
    p.add_argument("--limit", type=int, default=None, help="Run at most N configs (after filtering).")
    p.add_argument("--dry-run", action="store_true", help="Print the commands without running.")
    opts = p.parse_args()

    os.makedirs(opts.results_dir, exist_ok=True)
    if os.path.exists(opts.configs):
        with open(opts.configs) as f:
            all_configs = json.load(f)
    else:
        all_configs = generate_configs(opts.configs, opts.lesions, opts.axes,
                                        opts.baselines, opts.seeds)

    # apply subset filters
    cfgs = [c for c in all_configs
            if c["lesion"] in opts.lesions and c["error_axis"] in opts.axes
            and c["baseline"] in opts.baselines and c["seed"] in opts.seeds
            and (opts.severities is None or c["severity"] in opts.severities)]
    if opts.limit:
        cfgs = cfgs[:opts.limit]

    runnable = sum(1 for c in cfgs if c["baseline"] in RUNNABLE_BASELINES)
    print("=" * 64)
    print(f"B-sweep: {len(cfgs)} configs selected | {runnable} runnable now "
          f"({','.join(sorted(RUNNABLE_BASELINES))}), {len(cfgs) - runnable} will be skipped")
    print(f"max_epochs={opts.max_epochs} workers={opts.workers} "
          f"{'DRY RUN' if opts.dry_run else ''}")
    print("=" * 64)

    print(f"devices: {opts.devices}")
    worker = functools.partial(execute_single_run, opts=opts)
    counts = {}
    if opts.workers == 1:
        global _WORKER_DEVICE
        _WORKER_DEVICE = opts.devices[0]
        for i, c in enumerate(cfgs, 1):
            r = worker(c)
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            if i % 10 == 0 or i == len(cfgs):
                print(f"[{i}/{len(cfgs)}] " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    else:
        if opts.workers > len(opts.devices) and opts.devices != ["cpu"]:
            print(f"NOTE: {opts.workers} workers on {len(opts.devices)} device(s) -- "
                  f"some GPUs will host multiple concurrent runs.")
        counter = mp.Value("i", 0)
        with mp.Pool(processes=opts.workers, initializer=_init_worker,
                     initargs=(opts.devices, counter)) as pool:
            for i, r in enumerate(pool.imap_unordered(worker, cfgs), 1):
                counts[r["status"]] = counts.get(r["status"], 0) + 1
                if i % 10 == 0 or i == len(cfgs):
                    print(f"[{i}/{len(cfgs)}] " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    print("=" * 64)
    print("Done. Status:", "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"Per-run results + metrics in '{os.path.abspath(opts.results_dir)}'")
    print("=" * 64)


if __name__ == "__main__":
    main()