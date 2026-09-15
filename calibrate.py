#!/usr/bin/env python3
"""
calibrate.py: compute the --stim-ua-per-unit value that makes damage_frac meaningful.

The Shannon damage threshold is uncalibrated by default, so damage_frac reads ~0 for
everyone. Calibrating puts the threshold at the *uncautious* CPN's p99 stim amplitude, so
vanilla sits ~1% at threshold and more cautious settings show lower nonzero damage.

How it works:
  1. Run a vanilla B1 cell (uncalibrated) of the lesion being ablated; its metrics report "amp_p99".
  2. python calibrate.py <metrics.json> [more jsons ...]   (mean over several seeds)
  3. Pass the printed value to that lesion's runs (--stim-ua-per-unit), one invocation per lesion.

Handles all metrics layouts:
  - stage1 summary keyed by lesion:  {"f5m1": {..., "amp_p99": ...}}
  - per-run result record:           {"lesion": ..., "metrics": {..., "amp_p99": ...}}
  - flat metrics:                    {..., "amp_p99": ...}
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safety import StimCalibration


def amp_p99_from(path):
    rec = json.load(open(path))
    candidates = []                                   # (lesion, metrics_dict)
    if isinstance(rec.get("metrics"), dict):          # per-run result record
        candidates.append((rec.get("lesion"), rec["metrics"]))
    candidates.append((rec.get("lesion"), rec))       # flat
    for k, v in rec.items():                          # lesion-keyed summary
        if isinstance(v, dict) and "amp_p99" in v:
            candidates.append((k, v))
    for lesion, m in candidates:
        if isinstance(m, dict) and m.get("amp_p99") is not None:
            return float(m["amp_p99"]), rec.get("baseline"), (lesion or rec.get("lesion"))
    raise SystemExit(f"{path}: no 'amp_p99' found in any layout. Re-run with the updated "
                     f"stage1 on a completed B1 run.")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python calibrate.py <metrics.json> [more jsons ...]")
    p99s, base, lesion = [], None, None
    for path in sys.argv[1:]:
        v, base, lesion = amp_p99_from(path); p99s.append(v)
    ref = sum(p99s) / len(p99s)
    ua = StimCalibration.calibrate_ua_per_unit(reference_amp=ref).ua_per_unit
    print(f"lesion             : {lesion}")
    print(f"baseline (ref)     : {base}   (should be B1 / uncautious)")
    print(f"amp_p99 (reference): {ref:.6g}   (mean of {len(p99s)} run(s))")
    print()
    print(f"--stim-ua-per-unit {ua:.6g}")
    if base and base != "B1":
        print(f"\nWARNING: reference is {base}, not B1. Calibrate off the *uncautious* baseline.")


if __name__ == "__main__":
    main()