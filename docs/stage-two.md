# Stage 2 manual

This page is the complete operational reference for `stage2_ensemble.py`. Where Stage 1 runs one training job under one error, Stage 2 builds an **ensemble** of `M` emulators and reports both how each one recovers and how much the trained emulators disagree. This disagreement is the raw epistemic-uncertainty signal the trust-aware objective of Stage 3 consumes.

## What the harness does

`stage2_ensemble.py` trains `M` emulators - each from a different random seed - and (optionally) a different randomly-drawn error configuration, after which it summarises the ensemble members' runs. It reuses the Stage 0 and Stage 1 helpers so the behavior matches those harnesses on overlap.

The reason each member is trained by running the loop once per member is worth mentioning: this code trains the emulator **inside** the experiment loop, and there is no standalone emulator trainer. So the honest way to produce `M` trained emulators without modifying the training loop is to run that loop `M` times. Wiring the ensemble in as a single emulator inside one loop is the Stage 3 integration: this script is the Stage 2 deliverable that it builds on. 
One practical consequence is that each member run also trains a co-processor - which, unfortunately, is some wasted compute - so always cap runs with `--max-epochs`.

## Building the ensemble

The central knob is **`--num-members`** (`M`), which defaults to `5`. Each member gets its own initialization seed (`--seed` plus the member index), and **`--member-errors`** decides how each member's error configuration is drawn.

- **`none`** (default) leaves every member clean, so members differ only by their initialization. This is the standard deep ensemble.
- **`random`** draws a different configuration per member, independently choosing among clean, parametric, and structural, and (when the state-hook wiring is present) distributional. The drawn values match the error definitions exactly: parametric and distributional magnitudes are sampled from the discrete levels `{0.05, 0.1, 0.2, 0.4}`; structural members distort **either** capacity, drawn from `{20, 40, 160}` units, **or** topology, drawn from `{gru, rnn}`. These sets deliberately exclude the correct defaults (87 units and the LSTM), so a randomly-drawn structural member always carries a real mismatch rather than accidentally being error-free.
- **`parametric`**, **`structural`**, and **`distributional`** apply the same fixed configuration to every member, using `--sigma-p`, `--en-num-neurons` / `--en-cell-type`, or `--drift-delta` respectively.

**NOTE**: parametric and structural errors diversify the emulator itself, through its weights or architecture, while distributional error drifts the brain the ensemble member trains against. A distributional member thus ends up being a correct emulator of a drifted brain rather than a wrong emulator of the true brain, which is a legitimate but conceptually different source of diversity worth remembering while interpreting the ensemble.

## Flag reference

- **`--num-members`** sets `M`, the number of emulators to train (default `5`).
- **`--member-errors {none,random,parametric,structural,distributional}`** selects how each member's configuration is chosen (default `none`).
- **`--sigma-p`**, **`--en-num-neurons`**, **`--en-cell-type`**, **`--drift-delta`**, **`--drift-ramp`** supply magnitudes and the distributional timescale, with the same meanings as in Stage 1. The `--drift-ramp` value is a single global timescale for the whole ensemble rather than a per-member draw.
- **`--seed`** is the base seed; member `i` uses `--seed + i` for both its initialization and its random configuration draw, so an entire ensemble is reproducible.
- **`--lesion {f5m1,aip,m1}`**, **`--device`**, **`--log-dir`**, **`--recover`**, **`--coadapt`**, **`--stim-sigma`**, and **`--en-ready-floor`** behave as in the earlier stages. Stage 2 trains one ensemble for one lesion, so it does not accept `all`.
- **`--stop-pct`** and **`--max-epochs`** set the per-member stopping rule. The Stage 1 caveat applies to every member: injected error can hold true recovery below `--stop-pct`, so `--max-epochs` is strongly recommended, and the harness warns if an error mode is selected without it.
- **`--baseline-only`** builds each member and prints its baselines without training, which is the fast way to confirm the ensemble is configured as you intended.

## Understanding the output

The harness prints a header for each member with its seed and drawn configuration, then that member's best recovery. The final summary lists every member's recovery and configuration, the mean, minimum, maximum, and spread of recovery across members, and a single disagreement number.

The disagreement number is the variance across the trained emulators' predictions on a common probe input, averaged over the output dimensions. It is a diversity proxy instead of a calibrated uncertainty: since the probe is a random input rather than actual brain states, it measures how differently the emulators respond to random inputs, not how much they disagree where it matters. We should thus treat it as a sanity signal that the ensemble is genuinely diverse. The calibrated version, disagreement evaluated on real rollout inputs, etc. is delivered with Stage 3.

## Relationship to Stage 3

The emulators here are single-headed and trained with the repository's mean-squared-error objective, so the uncertainty available at this stage is disagreement across members. The per-member predicted **variance** used by the trust-aware loss comes into play only once the ensemble is wired in as the emulator through the Stage 3 adapter. The two stages connect cleanly: Stage 2 proves the members train and disagree, and Stage 3 turns that disagreement into the trust-aware penalty.

## Examples of usage

```
python stage2_ensemble.py --num-members 5 --lesion f5m1 --max-epochs 2000
python stage2_ensemble.py --num-members 5 --member-errors random --max-epochs 2000
python stage2_ensemble.py --num-members 3 --member-errors structural --en-cell-type gru --max-epochs 2000
python stage2_ensemble.py --num-members 5 --member-errors distributional --drift-delta 0.2 --max-epochs 2000
python stage2_ensemble.py --num-members 5 --baseline-only
```
