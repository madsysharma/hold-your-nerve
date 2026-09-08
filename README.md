# HOLD YOUR NERVE: ROBUST AND SAFE CLOSED-LOOP STIMULATION BY NEURAL CO-PROCESSORS UNDER FLAWED EMULATOR MODELS

This repository contains the implementation of our framework for trust-aware neural co-processors, built on top of the Bryan et al. neural co-processor testbed (`https://github.com/mmattb/coproc-poc`). The purpose of the code is to create an entirely simulation-based framework to demonstrate safe, closed-loop neurostimulation, all through the following features:
- A benchmark to inject controlled, parameterized emulator error along three unique axes - parametric (the right model fit badly), distributional (the brain drifts away from the model, and is calibrated to real implant drift), and structural (the model class itself is misspecified) - and measure task recovery and safety cost together rather than separately.
- A method called trust-aware CPN training, using an ensemble of emulators supplying epistemic uncertainty through their disagreement, a MOPO-style penalty discouraging the policy from relying on regions where that disagreement is high and an optional conformal layer capping stimulation magnitude when emulator confidence falls.
- An insight for the benchmark to deliver, ie: model-exploitation gap, which is the difference between what the system believes it achieved and what it actually achieved. Our central hypothesis would be that the existing co-processor fails silently: its self-reported recovery holds flat while its true recovery and safety both degrade underneath.

## What's in this repo?
 
- A **simulated brain**, which is a modular recurrent neural network (mRNN) obtained from the code by Michaels et al. (available at: `https://github.com/JonathanAMichaels/hfopt-matlab`) It has three modules - M1, F5 and AIP - with 100 neurons each.
- A set of **lesions** - which knocks out either a set of neuron outputs or a set of inter-module connections - and which makes the simulated brain perform the reaching-and-grasping task badly.
- A **co-processor** made of two learned components that train in alternation:
  - The **CPN** (Co-Processor Network), which reads a low-resolution observation of the brain and decides what stimulation to inject.
  - The **EN** (Emulator Network), which learns to predict what the brain will do in response to stimulation, and which acts as the differentiable stand-in that lets gradients reach the CPN.
- A **stimulator** and an **observer**, which model the physical limits of a real neural interface. The observer only sees a blurred, 20-dimensional summary per module, and the stimulator can only drive 16 spatially broad channels.
- **`stage0_repro.py`**, a command-line entry point that displays the healthy and lesioned baselines, runs training and reports the recovery percentage, to verify that the original baselines are being met.

## Quick start

### Requirements

Make sure you have all the necessary libraries by running:
```bash
pip install --no-cache-dir numpy pandas torch attrs
```
 
The quickest way to confirm that your environment and data are correct is to compute the baselines without training anything:
 
```bash
python stage0_repro.py --lesion all --device cpu --baseline-only
```
 
It should take a couple of minutes and print a healthy loss, a lesioned loss and the headroom between them for each of the three lesions. If the lesioned loss is not clearly worse than the healthy loss, something is wrong with your setup, and the recovery numbers from a real run would be meaningless.
 
A full training run for a single lesion looks like this:
 
```bash
python stage0_repro.py --lesion f5m1 --device 0 --stop-pct 0.70 --max-epochs 30000
```
 
Be aware that training is genuinely slow. Runs of hundreds of thousands of epochs are normal for the harder lesions, so always set `--max-epochs` unless you intend to leave the job running for a long time.
 
## Credits
 
- The testbed, the co-processor design, and most of the code under `experiment/` originate from the public `coproc-poc` repository by Bryan et al.
- The simulated brain weights and the task dataset in `michaels/` were provided by Jonathan Michaels and are described in his 2020 PNAS paper on goal-driven modular neural networks. 

## Team Members

Heba, Madhuri, Ethan and Thejo. Agatha (mentor), Richard and Sean (PIs).
