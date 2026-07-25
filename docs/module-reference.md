# Module reference

This page describes each file in the repository, so you can find the right place to make a change. Files are grouped by role instead of being listed in alphabetical order.

## Entry points

**`stage0_repro.py`** is the command-line harness for Stage 0. It parses arguments, resolves the device, builds the appropriate configuration through the upstream builders, prints baselines, runs training and reports recovery against the gate. It deliberately avoids re-implementing any upstream logic, and overrides upstream behavior only by setting module-level constants such as `cpn_epoch_cpn.STOP_PCT` and `cpn_epoch_cpn.MAX_EPOCHS` before training begins. The uniform `GATE = 0.70` constant lives near the top of this file.

**`run_nocoadap_phase1.py`** and **`run_nocoadap_phase2.py`** are the original upstream driver scripts, kept for reference. They contain hard-coded absolute paths from the original author's machine, including a log directory under `/home/mbryan/`, so they will not run unmodified.

**`train.py`** is explicitly marked as an old file in its own first line. It imports modules such as `stats` and `utils` from the top level rather than from the `experiment` package, so those imports no longer resolve. Think of it as historical context rather than working code.

## The co-processor

**`cpn.py`** defines `CPN_EN_CoProc`, which is the state machine coordinating the whole co-processor. It owns both the CPN and the EN, holds the current epoch type, routes `forward` and `feedback` calls to whichever phase handler is active, and decides in `finish` whether to switch phases. It also implements `train_en_closed_loop`, which is the offline replay path that re-trains the emulator on saved data without re-running the brain. Checkpoint saving happens here in `report`, every 50 epochs.

**`cpn_epoch_en.py`** implements the emulator-training phase. Its `finish` method contains the readiness logic that decides when to hand control to the CPN, together with the learning-rate schedule keyed on validation prediction loss and the reset-on-divergence behavior. If you want to change when the emulator is considered trustworthy, this is the file.

**`cpn_epoch_cpn.py`** implements the policy-training phase. It holds the module-level `STOP_PCT` and `MAX_EPOCHS` constants that the harness overrides, four alternative learning-rate schedules of which two are actually selected, the per-lesion CPN phase length in `cpn_epoch_max_len`, and the logic deciding when to hand control back to the emulator. It also contains a local patch guarding against a `None` recent task loss, described on the troubleshooting page.

**`cpn_model.py`** defines the CPN architectures. `CPNModelLSTM` is the one actually used. `CPNNoiseyLSTMCollection` is the noisy variant used during emulator training, which perturbs parameters per batch element so the emulator sees a wide range of policy behaviors rather than only the current one.

**`stim_model.py`** defines the emulator architectures and the `get_stim_model` factory that builds an emulator together with its AdamW optimiser. `StimModelLSTM` is the one actually used.

**`cpn_utils.py`** is small but load-bearing. It defines the `EpochType` enumeration that names every phase, the `CPNENStats` record that carries per-epoch statistics into the logs, and the two loss functions. Note that `calc_pred_loss` compares predictions against the brain's actual outputs while `calc_train_loss` compares them against dataset targets, which is precisely the distinction between emulator accuracy and task performance.

## The testbed

Everything under `experiment/` comes from the upstream repository.

**`experiment/experiment.py`** is the core driver. It defines the `CoProc` interface that any co-processor must implement, the three configuration builders `get_config`, `get_aip_lesion_config`, and `get_m1_lesion_config`, and the `Experiment` class whose `run` method is the outer training loop. It also computes the healthy and lesioned baselines during staging, which is why `--baseline-only` can report them without training.

**`experiment/config.py`** defines the frozen `Config` dataclass and the single `get` function that every lesion builder funnels through. All the default constants live here, including 100 neurons per module, 20 observation dimensions, an observation sigma of 1.75, 16 stimulation channels, a stimulation sigma of 2.175, a stimulation decay of 0.3, and a 20 percent holdout. Change a default here and it changes everywhere.

**`experiment/mRNN.py`** contains `MichaelsRNN`, the simulated brain, and `MichaelsDataset`, the task data. The brain's `forward` applies input, recurrence, bias, and stimulation, then applies the lesion at the appropriate point depending on lesion type. Its `observe` method produces the per-module observation tuple that the co-processor receives, optionally dropping a module or masking neurons.

**`experiment/lesion.py`** defines the lesion classes and the `LesionType` enumeration. `LesionOutputsByIdxs` and `LesionConnectionsByIdxs` are the two used by the three standard conditions. Note that `LesionOutputs` samples its killed neurons randomly, unlike the index-based version, so it would not be deterministic across runs.

**`experiment/observer.py`** defines the observation models. `ObserverGaussian1d` is the standard one. `ObserverGaussian1dDrifting` adds slow random-walk bias to model electrode drift over time, and it is the reason the co-processor checkpoints observer state alongside network weights.

**`experiment/stim.py`** defines the stimulation models. `StimulusGaussianExp` is the standard one, combining Gaussian spatial spread with exponential temporal decay, and it maintains an internal accumulator so injected stimulation persists across timesteps.

**`experiment/stats.py`** defines `LossHistory`, which is the central bookkeeping object threaded through everything. It computes the recovery percentage, tracks the running maximum in `max_pct_recov`, and renders the per-epoch log lines. If you want to change what gets logged or how recovery is defined, this is the file.

**`experiment/utils.py`** holds shared helpers, including the hand-crafted `LSTMModel` that both the CPN and EN are built on, the Gaussian array-weight construction used by both the observer and the stimulator, the `ReTanh` activation, and `trunc_to_trial_end`, which applies the variable-trial-length mask.

**`experiment/michaels_load.py`** resolves the path to the bundled MATLAB file and loads it through SciPy. Its base directory is a relative path, which is why every command must be run from the repository root.

**`experiment/activation.py`** and **`experiment/class_stdevs.py`** are small support modules. The first maps activation names to classes, and the second computes the between-class and within-class variance statistics used for the class-separation metric in the logs.

## Data and artifacts

**`michaels/`** contains the fitted brain weights and the task dataset, supplied by Jonathan Michaels and described in his 2020 PNAS paper. The single 13 MB `.mat` file is required for everything.

**`experiment/recovered/`** contains the one pre-recovered brain checkpoint, which is for the F5-M1 connection lesion. Its filename encodes the lesion specification, and the absence of equivalent files for the AIP and M1 conditions is exactly why `--recover` is rejected for those.

## Things to know before you make changes

- The `Config` object is frozen, so you cannot mutate it after construction. Pass overrides through the builder functions instead.
- The harness controls upstream behavior by assigning to module-level constants before training starts. This works but it is global mutable state, so it only behaves correctly because the harness sets values per lesion inside `run_one` rather than once in `main`.
- The CPN phase length is not exposed as a flag and must be edited in `cpn_epoch_cpn.py` if you want shorter smoke tests.