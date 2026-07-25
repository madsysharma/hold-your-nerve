# Architecture

This page explains how the different components fit together. It is meant for someone who wants to understand or modify the system, and not just run it.


## The problem being modeled

<img width="3437" height="3048" alt="Frame 7" src="https://github.com/user-attachments/assets/1a265f1a-c620-45bd-a6d2-44dc33235c02" />


A neural co-processor is an implant that reads brain activity, decides a pattern of electrical stimulation and writes that stimulation back into the brain in order to restore lost function after injury, such as a stroke and the likes. The leading implementation of neural co-processors  (by Bryan et. al., 2023) involves a Co-processor Network (CPN) that trains its stimulation policy by backpropagating behavioral error through a pre-trained Emulator Network (EN), which is a learned model of how the brain responds to stimulation.

And that's the hard part: you cannot backpropagate through a real brain, so you cannot train the stimulation policy directly. The policy’s actions are electrical signals delivered into cortex: and a policy exploiting an emulator’s blind spot will confidently emit stimulation that the emulator predicts is therapeutic and that real brain tissue may experience as harmful.

So we need to train the emulator to accurately predict how the brain responds to stimulation. Once that desired level of accuracy is achieved, the emulator serves as a differentiable stand-in, and gradients from the task loss can flow through it to the stimulation policy. Because the brain and the policy both keep changing, the two networks have to be trained in alternation rather than once each.

Everything in this repository is a simulation of that setup, which means the ground-truth brain is available for evaluation even though the co-processor is never allowed to use it directly.

## The simulated brain

The brain is `MichaelsRNN` in `experiment/mRNN.py`. Its main properties are:

- **Three modules** named M1, F5, and AIP, with **100 neurons each**, giving 300 neurons in total. Module index 0 is M1, index 1 is F5 and index 2 is AIP.
- It is **initialised from fitted weights** in the bundled MATLAB file rather than trained from scratch: so it starts out already performing the task well.
- Its **input** at each timestep is a set of CNN-derived visual features describing the object to be grasped, along with a hold signal.
- Its **output** is 50 muscle activation values that are read out through a linear layer seeing only the M1 module. Output dimensions of >= 32 correspond to hand muscles.
- Its **dynamics** are leaky and continuous-time in nature, updating the state as `x + tdx/10` at each step, where `tdx` combines recurrence, input, bias, and any injected stimulation.

The task itself is a reaching-and-grasping dataset with variable-length trials. Trials are padded to a common length and masked by a `trial_end` indicator, so all the loss functions truncate predictions against that mask before scoring.

## The lesions

A lesion is an object that gets applied inside the brain's forward pass. Two kinds are used and they attach at different points: this matters when you reason about what the CPN can compensate for.

- **Output lesions** (`LesionOutputsByIdxs`) zero out a contiguous range of neuron activations. They are applied to the pre-activation response on every step. The AIP condition zeroes neurons 200 to 250, while the M1 condition zeroes neurons 0 to 50.
- **Connection lesions** (`LesionConnectionsByIdxs`) zero out blocks of the recurrent weight matrix, severing communication between modules. The default F5-M1 condition cuts F5 to M1, M1 to F5, AIP to M1, and M1 to AIP.

## The neural interface

Two objects model the physical limits of the implant: making them the reason the problem is non-trivial.

The **observer** (`ObserverGaussian1d`) is what the co-processor is allowed to see. It applies 20 Gaussian-weighted read-out kernels with a sigma of 1.75 across each module's 100 neurons, so the co-processor receives a blurred 20-dimensional summary per module rather than raw activity. This reflects the fact that an electrode array measures a spatial average and not individual neurons.

The **stimulator** (`StimulusGaussianExp`) is what the co-processor is allowed to write. It has 16 channels by default. Each channel spreads its effect spatially with a Gaussian of sigma 2.175, and each injection decays exponentially over time with a decay rate of 0.3, so stimulation persists and accumulates rather than acting instantly. Critically, the stimulator only covers a padded window of the 300 neurons, and that window is configured per lesion.

The stimulation window placement is worth understanding, since it drives most of the difficulty differences between conditions:

- For the **F5-M1** lesion, stimulation covers neurons 0 to 100, which is the M1 module.
- For the **AIP** lesion, stimulation covers neurons 0 to 100 while the damage is at neurons 200 to 250. The stimulator and the lesion do not overlap at all.
- For the **M1** lesion, stimulation covers neurons 50 to 100 while the damage is at neurons 0 to 50. Like before, there is no overlap.

For both AIP and M1, the co-processor cannot stimulate the damaged neurons directly and must act indirectly through recurrent connectivity. This aligns with real-life scenarios, and it is why those two conditions recover far more slowly than the F5-M1 condition.

## The two learned networks

The **CPN** (Co-Processor Network) is the source of the stimulation policy. It is an Long Short-Term Memory (LSTM) defined in `cpn_model.py`, which takes the concatenated observations from all observed modules plus the trial-end flag, and outputs a 16-dimensional stimulation vector with one value per channel. Its input dimension depends on how many modules are observed: 61 when all three modules are visible, and 41 for the AIP and M1 configurations, both of which drop the M1 observation.

The **EN** (Emulator Network) is the differentiable brain surrogate. It is also an LSTM, defined in `stim_model.py`, taking the M1 observation, the current stimulation vector and the trial-end flag, giving 37 input dimensions, and predicting the brain's 50 muscle outputs for the next timestep.

Two unique loss functions drive these networks, and confusing them is the most common source of misunderstanding when reading the code:

- The **prediction loss** is the mean squared error (MSE) between the EN's predicted muscle outputs and the brain's actual muscle outputs. It measures emulator accuracy and trains the EN.
- The **task loss** is the MSE between the EN's predictions and the dataset's target muscle outputs. It measures task performance and trains the CPN.

## The alternating training loop

`CPN_EN_CoProc` in `cpn.py` is a state machine that switches between phases. The outer loop lives in `Experiment.run` in `experiment/experiment.py`, which steps the brain through a trial, hands each observation to the co-processor, receives a stimulation vector and then reports the results back.

The **EN phase** works as follows: 
- The CPN's parameters are frozen and stimulation is generated by a purposefully noisy copy of the CPN so the emulator sees a diverse range of inputs rather than only the current policy's narrow behavior. 
- The prediction loss is backpropagated into the EN alone. Every 10 epochs a validation epoch runs using the clean CPN, and the resulting validation prediction loss is what decides readiness.
- The EN is declared ready when its validation prediction loss falls below `max(0.02 * recent_train_loss, 0.0003)` and at least 100 epochs have passed, or when 3000 epochs have elapsed regardless. The second condition is a timeout, and runs that hand off mainly via the timeout are a sign that the readiness threshold is poorly matched to the condition. If the validation loss goes to NaN, infinity or above 1.5, the EN is discarded and rebuilt from scratch.

The **CPN phase** works as follows:
- It freezes the EN and unfreezes the CPN. 
- Stimulation is generated by the clean CPN, the EN predicts the outcomes and the task loss is backpropagated through the frozen EN into the CPN. 
- Control returns to the EN phase when the emulator has drifted out of trust, specifically when the prediction loss exceeds `max(task_loss / 10, 6e-4)` or when a maximum CPN-phase length is reached or when the recent task loss history shows the CPN regressing.

The CPN-phase length limit is hard-coded rather than exposed as a flag. It is 100 epochs for the M1 lesion and 200 epochs otherwise, and it lives in `CPNEpochCPN.__init__` in `cpn_epoch_cpn.py`. Shortening a smoke test thus requires changing that value directly.

There is also an **offline EN replay** path:
- When `recycle_thresh` is set - which the harness always does with a value of 1 - the co-processor saves the observations and stimulations from recent epochs and re-trains the EN on that stored data without re-running the brain.
- These epochs are labelled `EN_OFFLINE` in the logs, and in practice, they dominate the epoch counts by a big margin.

An important result is that only a small fraction of total epochs are actual CPN learning steps. So when you look at a slow-moving recovery curve, this is usually the explanation.

## Co-adaptation

When `coadapt` is enabled, the brain itself becomes trainable at a very low learning rate of 1e-7, and it takes a gradient step every fifth epoch. This mirrors the biological reality that the brain adapts to a chronic implant rather than staying the same. Gradients are masked differently depending on lesion type, using `set_connection_coadap_grads` for connection lesions and `set_end_to_end_coadap_grads` for output lesions.

The upstream defaults are inconsistent across conditions, which is a frequent source of confusion. The F5-M1 builder defaults co-adaptation off, while the AIP and M1 builders default it on. The harness exposes this through `--coadapt` with three modes so you can force a consistent choice or deliberately reproduce the upstream mix.

## Storing recovery metrics

Recovery is computed in `LossHistory.calc_pct_recov` in `experiment/stats.py`:

```
recovery = 1.0 - (task_loss - healthy_loss) / (lesioned_loss - healthy_loss)
```

A recovery of 0.0 means the co-processor achieved nothing beyond the unstimulated lesioned baseline, and 1.0 means it fully restored healthy performance. The reported figure is `max_pct_recov`, which is the best value seen at any point during the run rather than the final value. This matters when you interpret results, because a run that peaked early and then degraded will still report its peak.
