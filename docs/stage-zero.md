# Stage 0 manual

This page is the complete operational reference for `stage0_repro.py`. It covers every flag, the differences between the three lesion conditions and how to understand what gets printed.

## What the harness does

`stage0_repro.py` is a thin wrapper that calls the upstream repository's own configuration builders, staging code and training loop. It does not re-implement any of the science, and merely exists to:

- Build the correct `Config` object for a chosen lesion, applying any overrides you passed on the command line.
- Print the healthy and lesioned baselines so you can confirm the lesion actually degraded behavior.
- Run training and report the best recovery percentage achieved.
- Compare that recovery against a gate and print a pass or fail verdict.

The design principle throughout is that an unspecified flag leaves the repository default untouched, rather than pinning it to a harness-chosen value. This matters most for co-adaptation where the upstream defaults differ per lesion.

## Flag reference

### Choosing what to run

- **`--lesion {f5m1,aip,m1,all}`** picks the lesion condition, defaulting to `f5m1`. Passing `all` runs the three conditions sequentially in a single process, which is convenient but means a crash midway through can lose the later conditions.
- **`--device`** takes either `cpu` or a CUDA device index as a string (eg: `0`), defaulting to `cpu`. If you request a GPU and PyTorch was built without CUDA, the harness exits immediately with an explanatory message instead of failing later.
- **`--baseline-only`** builds everything, prints the baselines, and stops before training. This is the correct fast verification of a setup.
- **`--log-dir`** sets where model checkpoints are written, defaulting to `./models`. Checkpoints for the CPN, EN, ground truth brain and observer are saved every 50 epochs into a subdirectory named after the configuration string and a new UUID.

### Experimental conditions

- **`--coadapt {off,on,upstream}`** controls whether the simulated brain adapts alongside the co-processor, defaulting to `off`. The three modes behave differently in an important way:
  - `off` forces co-adaptation false for every lesion, which is the clean vanilla baseline and the one to use when you want all three conditions to share a single protocol.
  - `on` forces it true for every lesion.
  - `upstream` passes nothing and lets each builder use its own default, which is false for F5-M1 but true for AIP and M1. Use this when you specifically want to match upstream behavior, and be aware that it means the three conditions are not running the same protocol.
  - A bare `--coadapt` with no value means `on`, which preserves compatibility with older invocations.
- **`--recover`** loads a pre-recovered brain that has already adapted to the lesion. Right now, this only works for the F5-M1 condition since that is the only pre-recovered model bundled with the upstream repository, and the harness rejects it for AIP and M1. When you pass `--lesion all` together with `--recover`, the flag is silently skipped for the unsupported conditions rather than aborting the whole sweep.

### Actuation model

- **`--stim-sigma`** sets the spatial spread of each stimulation channel, defaulting to 2.175, which matches both the upstream repository and the corresponding published paper by Bryan et. al. Lowering it makes stimulation more focal.
- **`--num-stim-channels`** sets the number of virtual electrodes, with a repository default of 16. Adding channels tiles the neuron span more densely, which is the main lever for reducing sensitivity to a narrow sigma.
- **`--stim-decay`** sets the per-step temporal decay of injected stimulation, with a repository default of 0.3 (matching the upstream repository). A smaller value lets stimulation persist and integrate over time, which can partly compensate for a narrower spatial spread.

The last two flags change the physical actuation model rather than the learning setup. If you use them, report them as deliberate design choices in your methods section rather than presenting the result as a straight reproduction. The same warning applies to any `--stim-sigma` value other than 2.175.

### Scheduling and stopping

- **`--stop-pct`** stops the run once best recovery reaches this fraction, defaulting to 0.70. Upstream only stops at 0.99, so this flag exists to avoid pointless extra training once the gate is cleared.
- **`--max-epochs`** imposes a hard epoch cap so a non-converging run still terminates and reports its best recovery. There is no limit by default, which means an unattended run without this flag can continue for a very long time. Setting it is highly recommended.
- **`--en-ready-floor`** is intended to override the emulator readiness threshold, accepting either a single bare value applied to every lesion, per-lesion assignments such as `aip=0.0012,m1=0.0012`, or a mix of one bare fallback plus per-lesion overrides. **This flag currently has no effect on training.** See the troubleshooting page for details, because this is important to know before you spend a run on it.

## The lesion conditions

The conditions are not equally difficult, and treating them as interchangeable will lead you to wrong conclusions.

**F5-M1** is the default connection lesion, severing communication between the F5 and M1 modules along with two extra AIP-M1 links. Stimulation covers neurons 0 to 100 (the M1 module), so the co-processor is stimulating right where the readout happens. This condition does the most damage in absolute terms, with a lesioned loss near 0.0207, and it is the easiest to recover. Co-adaptation defaults to off upstream, and it is (currently) the only condition supporting `--recover`.

**AIP** is an output lesion zeroing neurons 200 to 250. Stimulation covers neurons 0 to 100, so it does not touch the damaged region at all and must act indirectly through recurrence. This lesion does the least damage, with a lesioned loss near 0.0045, which means the headroom is small and the recovery denominator is small too. Co-adaptation defaults to on upstream.

**M1** is an output lesion zeroing neurons 0 to 50. Stimulation covers neurons 50 to 100, so once again there is no overlap with the damage. The damage is severe, with a lesioned loss near 0.0211. This condition also gets special treatment inside the code: it uses a more aggressive learning-rate schedule and a shorter CPN phase of 100 epochs rather than 200. Co-adaptation defaults to on upstream.

## Understanding the gate

`stage0_repro.py` as committed uses a single uniform gate of 70 percent recovery, applied identically to all three lesions, and it only declares the overall Stage 0 gate passed when all three conditions clear it.

**A uniform 70 percent bar is not the right acceptance criterion for a faithful reproduction.** Bryan et al. report non-co-adaptive AIP recovery of roughly 15 percent, so a correct reproduction of that condition would fail a 70 percent gate. The appropriate criterion is a per-condition fidelity gate that checks your numbers against the published values for the same condition, for example requiring the lesioned loss to fall within one percent of the published value and the recovery to fall within ten percentage points of it.

If this project has already moved to a fidelity gate elsewhere, be aware that this copy of `stage0_repro.py` still contains the uniform version, and the `GATE = 0.70` constant near the top of the file along with the summary logic at the bottom of `main` are the places to change.

## Reading the output

Each condition prints a header block echoing the resolved settings, then the baselines, then training logs, then a result line. The final summary looks like this:

```
STAGE 0 SUMMARY
  coadapt=off  stim_sigma=1.75  stop_pct=0.7  max_epochs=30000
  f5m1 :   72.4%  -> PASS   (en_ready_floor=repo default)
  aip  :   17.2%  -> FAIL   (en_ready_floor=repo default)
```

The per-epoch training logs come from the root logger and interleave several fields. The most useful ones are `pct_recov`, which is the current recovery, `task_loss`, which is what the CPN is minimising, and `pred_loss` alongside `pred_val_loss`, which track emulator accuracy. Each line is also prefixed with the epoch type, so you can tell whether the system is in a CPN phase, an online EN phase, or an offline EN replay phase.

Expect the epoch-type mix to be heavily skewed toward EN and EN_OFFLINE. That is normal: but it also means that raw epoch counts are a poor proxy for how much the CPN has actually learned.

## Worked examples

Verifying a setup quickly on CPU without training:

```bash
python stage0_repro.py --lesion all --device cpu --baseline-only
```

Running a single condition with a sensible cap so it terminates:

```bash
python stage0_repro.py --lesion f5m1 --device 0 --max-epochs 30000 --stop-pct 0.70
```

Running all three conditions under one consistent protocol, which is what you want for a clean cross-condition comparison:

```bash
python stage0_repro.py --lesion all --device 0 --coadapt off --max-epochs 30000
```

Comparing co-adaptation on and off for a single condition:

```bash
python stage0_repro.py --lesion m1 --device 0 --coadapt off --max-epochs 30000
python stage0_repro.py --lesion m1 --device 0 --coadapt on  --max-epochs 30000
```

Sweeping the actuation model, which changes the physics rather than the learning setup:

```bash
python stage0_repro.py --lesion f5m1 --device 0 --stim-sigma 2.175 --num-stim-channels 32
python stage0_repro.py --lesion f5m1 --device 0 --stim-sigma 2.175 --stim-decay 0.15
```

## Practical tips on run length

Training is slow, and the harder conditions are very slow. A recorded AIP run at the paper's default sigma reached only about 17 percent recovery after roughly 508,000 epochs, with recovery gains arriving in bursts spaced thousands of epochs apart rather than climbing smoothly.

Some practical implications follow from this:

- Always set `--max-epochs` for unattended runs, so you get a reported number rather than an indefinite job.
- DO NOT conclude that a run has stalled from a few thousand flat epochs: because the burst pattern means flat stretches are expected.
- Because checkpoints are written every 50 epochs and the reported figure is the maximum recovery ever seen, an interrupted run still leaves usable artifacts on disk.