#!/usr/bin/env python3
"""
analysis.py -- post-hoc analysis and figures over the metrics produced by
run_experiments.py and run_ablations.py.

This is the third layer: run_experiments (main factorial) and run_ablations
(one-knob ablations) GENERATE per-run metrics; this file CONSUMES them and does
every statistical treatment and figure from the research doc that the logged data
already supports:

  covered here (data exists):
    - Delta vs. error-magnitude curves          [delta_curves]
    - safety-recovery Pareto frontier (lambda)  [pareto]
    - bootstrap 95% CIs per cell                [in the summary table]
    - Mann-Whitney U pairwise tests             [pairwise]
    - Holm-Bonferroni across the error-axis family

  NOT covered (need new eval-time logging first, so these emit an honest note
  instead of a fabricated figure):
    - reliability diagrams (uncertainty vs actual emulator error, per axis):
      needs (predicted_uncertainty, actual_error) logged per axis. The plotting
      helper already exists in uncertainty.plot_reliability_diagrams -- wire the
      logging, then call reliability_diagrams(...) below.
    - stimulation trajectory visualization: needs per-step state + uncertainty
      logged at eval.

Each run record (run_*.json) has: baseline, error_axis, severity, seed, lesion,
optional ta_lambda / ablation / ablation_value, status, and metrics =
{eta_true, eta_hat, delta, cvar, damage_frac, max_recov}.

Example:
  python analysis.py --results-dir results ablation_results --out-dir figures \
      --do delta_curves pareto pairwise
"""
import argparse
import glob
import json
import math
import os
import re
import random

try:
    from scipy.stats import mannwhitneyu
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")   # headless
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


METRIC_KEYS = ("eta_true", "eta_hat", "delta", "cvar", "damage_frac", "max_recov")


# ---------------------------------------------------------------- loading ----

def load_results(dirs):
    """Load every completed run record from one or more results dirs into flat rows."""
    rows = []
    for d in dirs:
        for path in glob.glob(os.path.join(d, "*.json")):
            if os.path.basename(path).endswith("_configs.json") or "configs" in os.path.basename(path):
                continue
            try:
                rec = json.load(open(path))
            except Exception:
                continue
            if not isinstance(rec, dict) or rec.get("status") != "completed":
                continue
            m = rec.get("metrics") or {}
            row = {k: rec.get(k) for k in
                   ("baseline", "error_axis", "severity", "seed", "lesion",
                    "ta_lambda", "ablation", "ablation_value")}
            for k in METRIC_KEYS:
                row[k] = m.get(k)
            rows.append(row)
    return rows


def severity_magnitude(severity):
    """Numeric magnitude for an axis level, or None for categorical (structural)."""
    if severity in (None, "clean"):
        return 0.0
    m = re.search(r"[-+]?\d*\.?\d+", str(severity))
    return float(m.group()) if m else None


# ------------------------------------------------------------- statistics ----

def bootstrap_ci(values, statistic=None, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for a statistic (default: mean)."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return (None, None, None)
    statistic = statistic or (lambda xs: sum(xs) / len(xs))
    point = statistic(vals)
    rng = random.Random(seed)
    boots = []
    n = len(vals)
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        boots.append(statistic(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return (point, lo, hi)


def holm_bonferroni(pvalues, alpha=0.05):
    """Holm-Bonferroni step-down. Returns list of (index, p, adjusted_alpha, reject)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    out = [None] * m
    still_rejecting = True
    for rank, i in enumerate(order):
        adj = alpha / (m - rank)
        reject = still_rejecting and pvalues[i] <= adj
        if not reject:
            still_rejecting = False
        out[i] = (i, pvalues[i], adj, reject)
    return out


def mann_whitney(a, b):
    """Two-sided Mann-Whitney U p-value, or None if scipy is unavailable / degenerate."""
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if not _HAVE_SCIPY or len(a) < 1 or len(b) < 1:
        return None
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return None   # all-identical groups etc.


# ------------------------------------------------------------- summaries ----

def summary_table(rows, metric="eta_true"):
    """Per-cell mean + bootstrap 95% CI for a metric, grouped by (baseline, axis, severity)."""
    cells = {}
    for r in rows:
        key = (r["baseline"], r["error_axis"], r["severity"])
        cells.setdefault(key, []).append(r.get(metric))
    table = []
    for key, vals in sorted(cells.items()):
        point, lo, hi = bootstrap_ci(vals)
        table.append({"baseline": key[0], "error_axis": key[1], "severity": key[2],
                      "metric": metric, "n": len([v for v in vals if v is not None]),
                      "mean": point, "ci_lo": lo, "ci_hi": hi})
    return table


def pairwise(rows, metric="cvar", axis_family=None, ref="B1", others=("B4", "B5")):
    """Mann-Whitney U of ref vs each other baseline per error axis, with
    Holm-Bonferroni across the error-axis family (as the doc specifies)."""
    axes = axis_family or sorted({r["error_axis"] for r in rows})
    tests = []
    for axis in axes:
        ref_vals = [r.get(metric) for r in rows if r["baseline"] == ref and r["error_axis"] == axis]
        for other in others:
            oth_vals = [r.get(metric) for r in rows if r["baseline"] == other and r["error_axis"] == axis]
            p = mann_whitney(ref_vals, oth_vals)
            tests.append({"axis": axis, "comparison": f"{ref} vs {other}",
                          "metric": metric, "p": p,
                          "n_ref": len([v for v in ref_vals if v is not None]),
                          "n_other": len([v for v in oth_vals if v is not None])})
    valid = [t for t in tests if t["p"] is not None]
    if valid:
        corrected = holm_bonferroni([t["p"] for t in valid])
        for t, c in zip(valid, corrected):
            t["holm_alpha"], t["reject"] = c[2], c[3]
    return tests


# --------------------------------------------------------------- figures ----

def delta_curves(rows, out_dir):
    """Delta vs error magnitude, one line per baseline, one panel per axis.
    This is the central figure: B1's Delta should climb, B4-B6 stay flat."""
    if not _HAVE_MPL:
        print("[delta_curves] matplotlib unavailable; skipping figure.")
        return
    axes = sorted({r["error_axis"] for r in rows if r["error_axis"] != "none"})
    if not axes:
        print("[delta_curves] no error-axis rows found.")
        return
    fig, panels = plt.subplots(1, len(axes), figsize=(5 * len(axes), 4), squeeze=False)
    for ax_i, axis in enumerate(axes):
        panel = panels[0][ax_i]
        baselines = sorted({r["baseline"] for r in rows})
        for base in baselines:
            pts = {}
            for r in rows:
                if r["baseline"] != base or r["error_axis"] != axis or r.get("delta") is None:
                    continue
                mag = severity_magnitude(r["severity"])
                if mag is None:
                    continue   # structural: not a magnitude scalar (handle categorically elsewhere)
                pts.setdefault(mag, []).append(r["delta"])
            if not pts:
                continue
            xs = sorted(pts)
            ys = [sum(pts[x]) / len(pts[x]) for x in xs]
            panel.plot(xs, ys, marker="o", label=base)
        panel.set_title(f"Delta vs magnitude: {axis}")
        panel.set_xlabel("error magnitude")
        panel.set_ylabel("exploitation gap Delta (eta_hat - eta)")
        panel.axhline(0, color="k", lw=0.5)
        panel.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "delta_vs_magnitude.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[delta_curves] wrote {path}")


def pareto(rows, out_dir, violation="cvar"):
    """Safety-recovery Pareto frontier from the lambda sweep: recovery vs violation.
    Uses rows that carry ta_lambda (the lambda-swept B4/B5 runs)."""
    if not _HAVE_MPL:
        print("[pareto] matplotlib unavailable; skipping figure.")
        return
    lam_rows = [r for r in rows if r.get("ta_lambda") is not None
                and r.get("eta_true") is not None and r.get(violation) is not None]
    if not lam_rows:
        print("[pareto] no lambda-swept rows (need run_experiments --ta-lambdas or the "
              "lambda ablation). Skipping.")
        return
    by_lam = {}
    for r in lam_rows:
        by_lam.setdefault(r["ta_lambda"], {"rec": [], "vio": []})
        by_lam[r["ta_lambda"]]["rec"].append(r["eta_true"])
        by_lam[r["ta_lambda"]]["vio"].append(r[violation])
    lams = sorted(by_lam)
    rec = [sum(by_lam[l]["rec"]) / len(by_lam[l]["rec"]) for l in lams]
    vio = [sum(by_lam[l]["vio"]) / len(by_lam[l]["vio"]) for l in lams]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(vio, rec, marker="o")
    for l, x, y in zip(lams, vio, rec):
        ax.annotate(f"λ={l}", (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"safety violation ({violation})")
    ax.set_ylabel("true recovery eta")
    ax.set_title("Safety-recovery Pareto frontier (lambda sweep)")
    fig.tight_layout()
    path = os.path.join(out_dir, "pareto_frontier.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[pareto] wrote {path}")


def reliability_diagrams(rows, out_dir, results_dirs):
    """H4 calibration figure. Reads the per-run *_reliability.json logs (written when a
    run used --log-reliability), groups the (uncertainty, emulator-error) pairs by the
    run's error axis, bins uncertainty into deciles, and plots mean emulator error per
    bin. A well-calibrated axis is monotone increasing; the pre-registered H4 prediction
    is that STRUCTURAL stays roughly FLAT -- the ensemble cannot detect it."""
    os.makedirs(out_dir, exist_ok=True)
    by_axis = {}
    for d in results_dirs:
        for rpath in glob.glob(os.path.join(d, "*_reliability.json")):
            run_id = os.path.basename(rpath)[:-len("_reliability.json")]
            recpath = os.path.join(d, run_id + ".json")
            axis = "unknown"
            if os.path.exists(recpath):
                try:
                    axis = json.load(open(recpath)).get("error_axis", "unknown")
                except Exception:
                    pass
            try:
                data = json.load(open(rpath))
            except Exception:
                continue
            b = by_axis.setdefault(axis, {"u": [], "err": []})
            b["u"] += data.get("u", [])
            b["err"] += data.get("err", [])
    if not by_axis:
        print("[reliability] no *_reliability.json logs found. Re-run the sweep with "
              "--log-reliability (run_experiments / run_ablations). This is the H4 figure.")
        return
    if not _HAVE_MPL:
        print("[reliability] matplotlib unavailable; skipping figure.")
        return
    axes = sorted(by_axis)
    fig, panels = plt.subplots(1, len(axes), figsize=(5 * len(axes), 4), squeeze=False)
    summary = {}
    for i, axis in enumerate(axes):
        panel = panels[0][i]
        pairs = sorted(zip(by_axis[axis]["u"], by_axis[axis]["err"]))
        n = len(pairs)
        if n < 10:
            panel.set_title(f"{axis}: too few points (n={n})")
            summary[axis] = {"n": n}
            continue
        nb = min(10, n)
        xs, ys = [], []
        for k in range(nb):
            chunk = pairs[k * n // nb:(k + 1) * n // nb]
            if chunk:
                xs.append(sum(p[0] for p in chunk) / len(chunk))
                ys.append(sum(p[1] for p in chunk) / len(chunk))
        panel.plot(xs, ys, marker="o")
        panel.set_title(f"reliability: {axis}")
        panel.set_xlabel("predicted uncertainty (binned)")
        panel.set_ylabel("mean emulator error")
        # rank correlation: does uncertainty track error? (flat ~ 0 -> ensemble blind)
        corr = mann_whitney is not None and None
        rho = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
        summary[axis] = {"n": n, "spearman_u_vs_err": rho}
        if rho is not None:
            panel.annotate(f"rho={rho:.2f}", (0.05, 0.9), xycoords="axes fraction", fontsize=9)
    fig.suptitle("Uncertainty calibration by error axis (H4): structural expected flat")
    fig.tight_layout()
    path = os.path.join(out_dir, "reliability_diagrams.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    json.dump(summary, open(os.path.join(out_dir, "reliability_summary.json"), "w"), indent=2)
    print(f"[reliability] wrote {path} + reliability_summary.json (per-axis rho: high on "
          f"parametric/distributional, ~0 on structural is the H4 result)")


def _spearman(a, b):
    """Spearman rank correlation, or None if degenerate / scipy absent."""
    if _HAVE_SCIPY and len(a) > 2:
        try:
            from scipy.stats import spearmanr
            r = spearmanr(a, b).correlation
            return None if (r != r) else float(r)
        except Exception:
            return None
    return None


def trajectories(rows, out_dir):
    print("[trajectories] NOT AVAILABLE: needs per-step state-space trajectory + "
          "uncertainty logged at eval. Not captured today.")


# ------------------------------------------------------------------ main ----

def main():
    p = argparse.ArgumentParser(description="Post-hoc analysis over sweep/ablation metrics.")
    p.add_argument("--results-dir", nargs="+", required=True,
                   help="One or more dirs of run_*.json (main sweep and/or ablations).")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--do", nargs="+",
                   default=["summary", "delta_curves", "pareto", "pairwise"],
                   choices=["summary", "delta_curves", "pareto", "pairwise",
                            "reliability", "trajectories"])
    p.add_argument("--metric", default="eta_true")
    p.add_argument("--violation", default="cvar", choices=["cvar", "damage_frac"])
    opts = p.parse_args()

    os.makedirs(opts.out_dir, exist_ok=True)
    rows = load_results(opts.results_dir)
    print(f"Loaded {len(rows)} completed run records from {opts.results_dir}")
    if not rows:
        print("Nothing to analyze (no completed runs found).")
        return

    if "summary" in opts.do:
        table = summary_table(rows, metric=opts.metric)
        out = os.path.join(opts.out_dir, f"summary_{opts.metric}.json")
        json.dump(table, open(out, "w"), indent=2)
        print(f"[summary] {len(table)} cells (mean + bootstrap 95% CI) -> {out}")

    if "pairwise" in opts.do:
        tests = pairwise(rows, metric=opts.violation)
        out = os.path.join(opts.out_dir, f"pairwise_{opts.violation}.json")
        json.dump(tests, open(out, "w"), indent=2)
        note = "" if _HAVE_SCIPY else "  (scipy missing -> p-values are null; pip install scipy)"
        print(f"[pairwise] Mann-Whitney U + Holm-Bonferroni on {opts.violation} -> {out}{note}")

    if "delta_curves" in opts.do:
        delta_curves(rows, opts.out_dir)
    if "pareto" in opts.do:
        pareto(rows, opts.out_dir, violation=opts.violation)
    if "reliability" in opts.do:
        reliability_diagrams(rows, opts.out_dir, opts.results_dir)
    if "trajectories" in opts.do:
        trajectories(rows, opts.out_dir)


if __name__ == "__main__":
    main()
