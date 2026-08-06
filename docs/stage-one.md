# Stage 1 manual

This page is the complete operational reference for `stage1_error_injection.py`. Where Stage 0 reproduces the clean testbed, Stage 1 injects one emulator error chosen on purpose and runs the same CPN-EN training: so that you can watch true recovery degrade and figure out the model-exploitation gap that motivates the whole project.

## What the harness does

`stage1_error_injection.py` has a similar structure to `stage0_repro.py` and reuses its helpers. For a chosen lesion it builds the repository `Config`, injects exactly one error axis and then runs the standard training loop, printing the healthy and lesioned baselines and the best recovery percentage achieved. Like in Stage 0, an unspecified flag leaves the repository default untouched: only the selected error axis is active and the other two remain turned off.

The main idea is that the chosen error corrupts the emulator (the "practice brain") while the co-processor is trained and evaluated against it. Since the co-processor is optimised through the corrupted emulator but scored on the true brain, a wrong emulator drags true recovery down: and the magnitude of that drop is the measurement we are looking for.

## The error axes

Exactly one axis is selected with `--error`, and its magnitude comes from the matching flag.

- **Parametric error** distorts the emulator's weights once, at the moment training hands off from emulator-fitting to co-processor-fitting. Each weight is updated as `w ← w + ε` with `ε ~ N(0, σ_p²·Var(w))`, so the noise standard deviation is `σ_p` times the weight's own standard deviation. The intended sweep is `σ_p ∈ {0, 0.05, 0.1, 0.2, 0.4}`, set with CLI argument `--sigma-p`.
- **Structural error** builds the emulator with a deliberately incorrect neuron capacity or topology. Capacity error changes the emulator's 87 units to `{20, 40, 160}` via `--en-num-neurons`, while topology error substitutes a GRU or a vanilla RNN for the LSTM via `--en-cell-type {lstm,gru,rnn}`. A third structural knob, ie: the wrong electrode-spread sigma, is adjustable through `--stim-sigma`.
- **Distributional error** applies a slow, low-dimensional rotation and translation to the brain's latent state at test time, so an emulator trained before the drift is evaluated after it. The magnitude is `δ` - set with `--drift-delta` - and both the low dimensionality and the slow timescale follow MINDFUL's description of real drift. This drift is injected as a forward hook on the true brain at the emulator-to-co-processor handoff, where the emulator is frozen on the pre-drift brain.

## Flag reference

### Picking the error

- **`--error {none,parametric,structural,distributional}`** selects the error axis, defaulting to `none` (a clean run identical to Stage 0). The harness refuses to start a parametric run without `--sigma-p > 0`, a distributional run without `--drift-delta > 0`, or a structural run without at least one of `--en-num-neurons` / `--en-cell-type`, so a wrongly specified error can never silently run as a clean baseline.
- **`--sigma-p`** sets the parametric magnitude and defaults to `0.0`.
- **`--en-num-neurons`** and **`--en-cell-type {lstm,gru,rnn}`** set the structural capacity and topology, leaving one unset keeps the repository default for that dimension.
- **`--drift-delta`** sets the distributional magnitude and defaults to `0.0`.

### Distributional timescale

- **`--drift-ramp`** controls how the drift develops over time and defaults to `-1`.
  - `-1` ramps slowly to full `δ` **within** one trial and restarts each trial, so the drift is confined to a session. The horizon is resolved automatically to the trial length.
  - `0` applies the full drift instantaneously at the handoff (a step change rather than a slow drift).
  - `N > 0` ramps slowly **across** trials over `N` total steps, so each trial starts more drifted than the last and the drift accumulates to `δ` and then holds. This is the multi-session MINDFUL model: we need to choose `N` to encode how many rollout steps correspond to MINDFUL's day-scale.
- **`--drift-k`** sets the dimensionality of the drift subspace (default being `8`), and **`--drift-seed`** fixes the drift direction (default `42`) so a run is easily reproducible.

### Shared knobs from Stage 0

These behave exactly as in Stage 0: **`--lesion {f5m1,aip,m1,all}`**, **`--device`**, **`--log-dir`**, **`--recover`** (F5-M1 only), **`--coadapt {off,on,upstream}`**, **`--stim-sigma`**, and **`--en-ready-floor`**.

### Stopping condition knobs

- **`--stop-pct`** stops a run once true recovery reaches this fraction, defaulting to `0.70`.
- **`--max-epochs`** caps the number of epochs and defaults to no cap.

## Stopping caveat to understand

The stop threshold is measured against **true** recovery on the (possibly drifted) brain, and injected error is specifically designed to hold true recovery below what a clean run reaches. If you set a `--stop-pct` that the error prevents you from reaching and you do not set `--max-epochs`, the run will train forever because its exit condition can never be met. The harness prints a warning whenever an error is injected without a cap.
The correct protocol for error runs is thus a fixed number of epochs: pass `--max-epochs` and read off the degraded recovery, then compare it against the clean `--error none` baseline at the same budget. That very difference is the model-exploitation gap.

## Understanding the output

For each lesion, the harness prints the healthy ceiling, the lesioned floor and the headroom between them, then trains and prints the best recovery. If the lesioned loss is not worse than the healthy loss, it warns that recovery will be meaningless. A `--baseline-only` run stops after the baselines: this is the fast way to confirm a setup before committing to training.

## Examples of usage

```
python stage1_error_injection.py --error parametric --sigma-p 0.2 --lesion f5m1 --max-epochs 2000
python stage1_error_injection.py --error structural --en-num-neurons 40 --lesion m1 --max-epochs 2000
python stage1_error_injection.py --error structural --en-cell-type gru --lesion aip --max-epochs 2000
python stage1_error_injection.py --error distributional --drift-delta 0.2 --lesion f5m1 --max-epochs 2000
python stage1_error_injection.py --error distributional --drift-delta 0.2 --drift-ramp 5000 --lesion f5m1 --max-epochs 2000
python stage1_error_injection.py --error none --baseline-only
```
