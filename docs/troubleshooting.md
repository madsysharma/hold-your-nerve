# Troubleshooting and known issues

This page covers the rough edges in the codebase, the errors you are most likely to run into and the behaviors that look like malfunctions but are normal.

## Known issues in the code

### `--en-ready-floor` currently has no effect

This is the most important item on this page, because the flag looks fully implemented and it is not.

`stage0_repro.py` parses the flag carefully, supports per-lesion values, sets `cpn_epoch_en.EN_READY_FLOOR` before each condition runs, and prints the resolved value in the header block. However, `cpn_epoch_en.py` never reads that attribute. The readiness decision in `CPNEpochEN.finish` uses a hard-coded expression instead:

```python
vl < max(0.02 * self.recent_train_loss, 0.0003)
```

The consequence is that the harness is setting an attribute that nothing consumes, so every run uses the hard-coded threshold regardless of what you pass. What's worse, the header prints the value you supplied, which makes it look as though the override took effect.

To actually wire it up, define a module-level constant in `cpn_epoch_en.py` and use it in the readiness comparison, for example by replacing the literal `0.0003` with `EN_READY_FLOOR`. Until that is done, avoid reporting any result as depending on this flag, and be aware that any earlier runs where you varied it were all using the same threshold.

### The `rtl is None` guard in `cpn_epoch_cpn.py`

`CPNEpochCPN.finish` contains a local patch that is not in the upstream repository:

```python
rtl = loss_history.recent_task_loss
if rtl is None:
    rtl = self.recent_train_loss
```

Without this guard, the code crashes with `unsupported operand type(s) for /: 'NoneType' and 'int'` when the emulator-to-policy handoff lands on a validation epoch before any CPN task loss has been recorded. This happens because validation epochs report through `report_val_last_result`, which passes `task_loss=None`, leaving `recent_task_loss` unset.

This is a genuine upstream bug rather than a problem introduced by the harness: and it is reachable using the upstream notebook's own `recycle_thresh=1` setting, which is exactly what the harness uses. **Keep the guard.** If you ever re-sync against upstream, re-apply it.

### Dead code that will not import

`train.py` is marked as old in its own first line and imports `stats`, `utils`, and `cpn_model` from the top level, but `stats` and `utils` now live inside the `experiment` package. It will raise an `ImportError` if you try to use it. It is kept for historical reference only.

`experiment.get_raw_data` contains a typo, calling a non-existent `config.get_data_raw` and writing `cuda=cuda**kwargs` where `cuda=cuda, **kwargs` was clearly intended. Nothing calls it, so it does no harm, but do not rely on it.

The `run_nocoadap_phase1.py` and `run_nocoadap_phase2.py` scripts contain hard-coded absolute paths from the original author's machine and will not run unmodified.

## Common errors

### `FileNotFoundError` on the `.mat` file

The data path is resolved relative to the current working directory. Run every command from the repository root. Confirm the file exists with:

```bash
ls michaels/Networks/M/recttanh-CNN-1e-3-1e-5-1e-1-1/recttanh-CNN-1e-3-1e-5-1e-1-1.mat
```

### `SystemExit` complaining that CUDA is unavailable

You requested a GPU device but PyTorch was installed as a CPU-only build. Either install a CUDA build or pass `--device cpu`. The harness checks this deliberately at startup so you find out immediately rather than partway into a long run.

### `--recover is not available for the aip lesion`

Only the F5-M1 condition has a pre-recovered brain checkpoint bundled with the repository, so `--recover` is rejected for the other two. If you pass `--lesion all --recover`, the flag is silently skipped for the unsupported conditions rather than aborting the sweep.

### `FileExistsError` when creating the log directory

Each run creates a fresh UUID-named subdirectory, so this should be rare. It can happen in the `dont_train` path, which uses a fixed `phase2` subdirectory name and calls `os.makedirs` without `exist_ok`. Remove or rename the existing directory.

### The lesion warning during baselines

If the harness prints a warning that the lesioned loss is not worse than the healthy loss, stop and investigate rather than proceeding. It means the lesion is not degrading behavior, which makes the recovery denominator meaningless or negative. Check that the lesion was constructed as expected before trusting anything downstream.

## Expected behaviors that are not bugs

**Most epochs are not CPN epochs.** Emulator training, both online and offline, dominates the epoch count. In one recorded AIP run, CPN epochs made up under four percent of roughly 508,000 total epochs. Raw epoch counts are therefore a poor proxy for how much policy learning has happened.

**Recovery improves in bursts rather than smoothly.** The same run showed gains arriving in bursts spaced roughly 3,200 epochs apart. A few thousand flat epochs does not indicate a stalled run.

**The AIP condition recovers poorly.** This is the published result, not a reproduction failure. The AIP lesion does the least damage, the stimulation window does not overlap the damaged neurons, and the published non-co-adaptive recovery is roughly 15 percent.

**The emulator is periodically thrown away.** When validation prediction loss becomes NaN, infinite, or exceeds 1.5, the emulator is rebuilt from scratch. This is intentional divergence recovery.

**Co-adaptation defaults differ between conditions.** Upstream defaults it off for F5-M1 and on for AIP and M1. If you want a consistent protocol across conditions, pass `--coadapt off` explicitly rather than relying on defaults.

## Things worth doing if you edit this code

- Wire up `EN_READY_FLOOR` so the flag does what it claims, or remove the flag so it stops advertising an override that does not happen.
- Replace the uniform gate with the per-condition fidelity gate, so a faithful reproduction is not reported as a failure.
- Expose `cpn_epoch_max_len` as a command-line flag, since editing the file is currently the only way to shorten a smoke test.
- Add a `requirements.txt` pinning PyTorch, NumPy, SciPy, and attrs, since the dependency list currently has to be reconstructed by reading imports.
- Add `__pycache__` to `.gitignore` and remove the committed bytecode, which currently exists for two different Python versions.
- Record the exact command line in the log directory alongside the checkpoints, so a saved run is self-describing rather than depending on shell history.