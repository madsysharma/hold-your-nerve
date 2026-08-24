"""
oracle.py -- Baseline B0 (oracle CPN): the EN is replaced entirely by the true mRNN.

This gives the upper bound on achievable recovery and isolates emulator error as the
single variable under study -- with a perfect emulator the model-exploitation gap
Delta = eta_hat - eta is zero by construction.

Why this needs its own path (unlike B2-B6, which are gated add-ons to the learned EN):
the testbed deliberately keeps the din-free, differentiable EN surrogate separate from
the din-driven true brain, and the brain's stimulus is DETACHED (stim.py) so it is a
non-differentiable environment. To let the CPN train THROUGH the true brain we (a) use a
frozen true-weight MichaelsRNN as the EN, (b) give it a retain-grad stimulus so gradients
flow to the stim, and (c) feed it din via a side-channel each step (the EN interface has
no din). Driven with the same reset/din/stim as `mike`, it reproduces the brain exactly
(verified) while remaining differentiable, so the CPN's loss backprops through the true
dynamics.
"""
import copy
import torch
import torch.nn as nn
from torch.optim import AdamW

import cpn
import cpn_epoch_cpn
import cpn_epoch_en
from experiment import mRNN, michaels_load


class OracleEN(nn.Module):
    """The true mRNN wired as the EN: frozen true weights, retain-grad stimulus, din via
    a side-channel. Reproduces `mike` step-for-step when driven identically."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        st = copy.deepcopy(cfg.stim_instance)
        if hasattr(st, "_retain_grad"):
            st._retain_grad = True          # differentiable stim so the CPN can train through
        self.brain = mRNN.MichaelsRNN(
            init_data_path=michaels_load.get_default_path(),
            stimulus=st, cuda=cfg.cuda)
        self.brain.set_lesion(cfg.lesion_instance)
        for p in self.brain.parameters():
            p.requires_grad = False         # the oracle is the TRUE brain: nothing to learn

        self.stim_dim = cfg.stim_dim
        self.obs_dim = cfg.observer_instance.out_dim
        self.in_dim = self.obs_dim + self.stim_dim + 1     # match the EN interface
        self.out_dim = self.brain.output_dim
        self.num_neurons = self.brain.num_neurons
        self.activation_func_t = torch.nn.Tanh
        # a single frozen dummy so AdamW(self.opt_en) has a (no-op) parameter list
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._next_din = None

    def set_next_din(self, din):
        """Called by the run loop before each step: the true brain's next task input."""
        self._next_din = din

    def init_step(self, din0):
        """Mirror the loop's initial `mike(din[:,0,:].T)` so the oracle starts in sync."""
        self.brain(din0)

    def forward(self, en_in):
        # extract the CPN's stim from en_in = cat(obs, stim, trial_end); obs is ignored
        # because the oracle carries the true brain's own state.
        stim = en_in[:, self.obs_dim:self.obs_dim + self.stim_dim]
        self.brain.stimulate(stim)
        if self._next_din is None:
            raise RuntimeError("OracleEN.forward before set_next_din(): the run loop must "
                               "feed din each step in oracle mode.")
        return self.brain(self._next_din)

    def reset(self):
        self.brain.reset()

    def parameters(self, *a, **k):
        return iter([self._dummy])          # frozen; EN 'training' is a no-op

    def named_parameters(self, *a, **k):
        return iter([("_dummy", self._dummy)])

    def rebuild_members(self):
        """Survive reset_en (like the ensemble/DR wrappers): fresh OracleEN + no-op opt."""
        new = OracleEN(self.cfg)
        return new, AdamW([new._dummy], lr=1e-3)


class OracleCoProc(cpn.CPN_EN_CoProc):
    """CoProc whose EN is the true mRNN (B0). EN 'training' is a no-op (the brain is
    perfect), so the run reaches CPN training almost immediately."""

    def __init__(self, cfg, log_dir=None, recycle_thresh=None):
        super().__init__(cfg, log_dir=log_dir, recycle_thresh=recycle_thresh)
        self.en = OracleEN(cfg)
        self.opt_en = AdamW([self.en._dummy], lr=1e-3)
        self.en_epoch = cpn_epoch_en.CPNEpochEN(
            self.en, self.opt_en, self.cpn, self.opt_cpn, self.cfg)
        self.cpn_epoch = cpn_epoch_cpn.CPNEpochCPN(
            self.en, self.opt_en, self.cpn, self.opt_cpn, self.cfg)
