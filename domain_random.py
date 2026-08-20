"""
domain_random.py: Baseline B2 (domain-randomized CPN).

The naive-robustness alternative to the uncertainty-aware ensemble: instead of
training the co-processor through one fixed wrong emulator, train it through a
distribution of wrong emulators. Each trial the CPN sees the EN with a freshly
resampled parametric perturbation (sigma_p ~ U(0, sigma_max)), so it cannot
overfit to one specific emulator's errors.

The perturbation is applied functionally (torch.func.functional_call), so the
stored EN weights are never modified: EN training proceeds on clean weights, and
randomization is gated to CPN epochs only. This is the "does brute-force robustness
match the trust-aware method?" control (H3's honest comparison point).
"""
import torch
import torch.nn as nn
from torch.func import functional_call
from torch.optim import AdamW

import cpn
import cpn_epoch_cpn
import cpn_epoch_en
from cpn_utils import EpochType
from error_injection import ErrorInjector


class DomainRandomizedEN(nn.Module):
    """Wraps an EN so each trial its forward uses a fresh random parametric
    perturbation, applied functionally (stored weights untouched). Gated by
    `randomize`: off -> clean base forward (used during EN training)."""

    def __init__(self, base_en, sigma_max=0.2, seed=0, cuda=None, cell_type="lstm",
                 randomize=True):
        super().__init__()
        self.base_en = base_en
        self.sigma_max = float(sigma_max)
        self.randomize = randomize
        self._seed = int(seed)
        self._cuda = cuda
        self.cell_type = cell_type   # from cfg, like new_en (base EN has no cell_type attr)
        self._gen = torch.Generator().manual_seed(self._seed)
        # interface attributes reset_en reads off the EN (base EN exposes these)
        for a in ("in_dim", "out_dim", "num_neurons", "activation_func_t"):
            if hasattr(base_en, a):
                setattr(self, a, getattr(base_en, a))
        self._perturbed = None
        self._resample()

    def _resample(self):
        """Draw a fresh sigma_p and build perturbed params = base + N(0,(sigma_p*std)^2)."""
        self._sigma_p = torch.rand((), generator=self._gen).item() * self.sigma_max
        pert = {}
        for name, p in self.base_en.named_parameters():
            std = p.detach().std()
            # Sample on the CPU generator (for reproducible seeding), then move to the
            # parameter's device BEFORE scaling: std lives on p.device, so multiplying
            # a CPU noise tensor by it would mismatch devices on GPU.
            noise = torch.randn(p.shape, generator=self._gen).to(p.device)
            pert[name] = p + noise * (self._sigma_p * std)   # in-graph wrt base params (frozen in CPN epochs)
        self._perturbed = pert

    def forward(self, x, *args, **kwargs):
        if self.randomize and self._perturbed is not None:
            return functional_call(self.base_en, self._perturbed, (x, *args), kwargs)
        return self.base_en(x, *args, **kwargs)

    def reset(self):
        if hasattr(self.base_en, "reset"):
            self.base_en.reset()
        self._resample()   # fresh perturbation each trial

    def parameters(self, *a, **k):
        return self.base_en.parameters(*a, **k)      # EN training optimises the BASE weights

    def named_parameters(self, *a, **k):
        return self.base_en.named_parameters(*a, **k)

    def rebuild_members(self):
        """Survive reset_en: fresh base EN, rewrapped (mirrors new_en's rebuild)."""
        new_base, _ = ErrorInjector.build_structural_en(
            self.in_dim, self.out_dim, num_neurons=self.num_neurons,
            activation=self.activation_func_t, cuda=self._cuda, cell_type=self.cell_type)
        new_wrapper = DomainRandomizedEN(new_base, self.sigma_max, seed=self._seed + 1,
                                         cuda=self._cuda, cell_type=self.cell_type,
                                         randomize=self.randomize)
        new_opt = AdamW(new_wrapper.parameters(), lr=1e-3, weight_decay=0.04)
        return new_wrapper, new_opt


class DomainRandomizedCoProc(cpn.CPN_EN_CoProc):
    """CoProc whose EN is wrapped in DomainRandomizedEN. Randomization is gated to
    CPN epochs (EN training sees the clean EN)."""

    def __init__(self, cfg, sigma_max=0.2, seed=0, log_dir=None, recycle_thresh=None):
        super().__init__(cfg, log_dir=log_dir, recycle_thresh=recycle_thresh)
        base_en = self.en
        self.en = DomainRandomizedEN(base_en, sigma_max=sigma_max, seed=seed,
                                     cuda=cfg.cuda, cell_type=cfg.en_cell_type)
        self.opt_en = AdamW(self.en.parameters(), lr=1e-3, weight_decay=0.04)
        self.en_epoch = cpn_epoch_en.CPNEpochEN(
            self.en, self.opt_en, self.cpn, self.opt_cpn, self.cfg)
        self.cpn_epoch = cpn_epoch_cpn.CPNEpochCPN(
            self.en, self.opt_en, self.cpn, self.opt_cpn, self.cfg)

    def forward(self, brain_data, loss_history, en_obs_override=None):
        # Perturb only during CPN epochs; EN training uses the clean EN.
        self.en.randomize = self.epoch_type in (EpochType.CPN, EpochType.VAL_CPN)
        return super().forward(brain_data, loss_history, en_obs_override=en_obs_override)