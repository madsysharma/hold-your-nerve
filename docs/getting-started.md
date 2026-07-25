# Getting started

This page takes you from a fresh checkout to a verified setup. It assumes no prior familiarity with the codebase.

## Requirements

- **Python 3.11** is the version the committed bytecode caches were built against, so it is the safest choice. Python 3.14 caches are also present in the tree, which suggests the code has been run under that version too, but 3.11 is the better-tested path.
- **PyTorch**, which is the only heavyweight dependency. If you intend to train on a GPU, install a CUDA build rather than the default CPU-only wheel.
- **NumPy**, **SciPy**, and **attrs**, all of which are imported directly by the codebase.

A minimal environment can be created like this:

```bash
python -m venv .venv
source .venv/bin/activate

# For CPU-only work:
pip install --no-cache-dir torch numpy scipy attrs

# For GPU work, install the CUDA build instead of the plain torch wheel:
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir numpy scipy attrs
```

There is no `requirements.txt` or `setup.py` in the repository, so the dependency list above is derived by reading the imports.

## Confirming the data is present

The simulated brain is not generated at runtime. It is loaded from a MATLAB file that ships with the repository, and everything fails immediately if that file is missing. Confirm it is there before doing anything else:

```bash
ls -lh michaels/Networks/M/recttanh-CNN-1e-3-1e-5-1e-1-1/recttanh-CNN-1e-3-1e-5-1e-1-1.mat
```

You should see a file of roughly 13 MB. This single file supplies both the pre-trained mRNN weights and the reaching-and-grasping task dataset used for training and validation.

## Working directory matters

The data path is resolved relative to the current working directory, because `experiment/michaels_load.py` builds it from the literal relative path `michaels/Networks`. In practice this means you must run every command from the repository root. Running any of the main scripts from a subdirectory, or from a wrapper script located elsewhere, will fail to find the data.

## Your first run

Start with the baseline-only mode, which builds every object, computes the healthy and lesioned reference losses and then stops before any training happens:

```bash
python stage0_repro.py --lesion all --device cpu --baseline-only
```

This is the correct smoke test because it exercises the data loading, the lesion construction, the observer and stimulator setup, and the loss computation, while costing only a couple of minutes.

### Reading the baseline output

For each lesion you will see a block that looks like this:

```
  Healthy ceiling  (MSE, no lesion)          : 0.000566
  Lesioned floor   (MSE, lesion, no stim)    : 0.020726
  Headroom (lesioned - healthy)              : 0.020160
  Hand-muscle ceiling / floor                : 0.000743 / 0.034133
```

These numbers indicate the following:

- The **healthy ceiling** is the task loss when the brain is undamaged and unstimulated. This is the best score any co-processor could hope to reach, so it serves as the oracle bound.
- The **lesioned floor** is the task loss with the lesion applied and no stimulation at all. This is the starting point that the co-processor has to improve on.
- The **headroom** is the gap between the two, and the denominator of the recovery percentage. A larger headroom means the lesion did more damage and thus, there is more to recover.
- The **hand-muscle** figures are the same two quantities computed over only the hand muscles, which have output dimensions >= 32. They are reported separately because grasping is the behaviorally unique part of the task.


A useful sanity check is that the AIP lesion does far less damage than the other two. That is expected, and is one reason the AIP condition recovers so poorly in absolute percentage terms.

## Moving to the GPU

Once the CPU baseline works, you can switch to a GPU for real training. The `--device` flag takes a CUDA device index as a string:

```bash
python stage0_repro.py --lesion f5m1 --device 0 --baseline-only
```

If PyTorch was installed without CUDA support, the harness detects this and exits with a clear message rather than failing deep inside the training loop. Training on CPU is possible but impractically slow for anything beyond a smoke test.

## What to read next

- If you want to understand what the system is doing, read [architecture.md](architecture.md).
- If you want to start running real experiments, read [stage0-manual.md](stage0-manual.md).