"""
trust_aware.py

Defines:
  1. trust_aware_loss(...): assembles the objective from the others' pieces: L = MSE(pred, target) + lam * u + beta * C_safety
  2. EnsembleENAdapter: the integration bridge. The CPN epoch currently does
     cur_pred = self.en(en_in) with a single EN.
     The adapter wraps an M-member ensemble so that same call returns the ensemble mean (so the CPN backprops through mu_bar), 
     while stashing the per-member outputs so the loss can read the uncertainty. 
     This is what lets the ensemble drop in with a one-line change at the call site.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from ensemble_types import EnsembleLike
from uncertainty import Estimator, uncertainty
from safety import SafetyWeights, safety_cost, soft_penalty, hard_halt


@dataclass
class TrustAwareConfig:
    """The knobs for Stage 3. 
    At integration time these become repo Config fields
    (lam, beta, ens_M, estimator, caution, ...), threaded like sigma_p/drift_delta from Stage 1."""
    lam: float = 1.0                       # Uncertainty-penalty weight
    beta: float = 1.0                      # Safety-cost weight
    estimator: Estimator = Estimator.MAX_FROBENIUS
    caution: str = "soft"                  # "soft" (MOPO) or "hard" (MOReL HALT)
    halt_threshold: float = 1.0            # Used when caution == "hard"
    halt_penalty: float = 10.0             # Used when caution == "hard"
    safety_weights: SafetyWeights = None


def trust_aware_loss(pred: torch.Tensor, target: torch.Tensor,
                     u: torch.Tensor, theta: torch.Tensor,
                     cfg: TrustAwareConfig):
    """Create the trust-aware objective for one batch/step.

    Args:
        pred:   ensemble mean prediction mu_bar, (batch, out_dim).
        target: behavioural target y*, (batch, out_dim).
        u:      per-batch uncertainty from uncertainty(), (batch,).
        theta:  stimulation applied, (batch, n_channels) - for the safety cost.
        cfg:    TrustAwareConfig.
    Returns:
        (total_loss scalar, parts dict) where parts breaks out MSE / uncertainty / safety for logging.
    """
    mse = F.mse_loss(pred, target)

    if cfg.caution == "soft":
        u_term = soft_penalty(u, cfg.lam).mean()
    elif cfg.caution == "hard":
        u_term = hard_halt(u, cfg.halt_threshold, cfg.halt_penalty).mean()
    else:
        raise ValueError(f"Unknown caution mode: {cfg.caution}")

    c_safety = safety_cost(theta, weights=cfg.safety_weights)
    s_term = cfg.beta * c_safety.mean()

    total = mse + u_term + s_term
    parts = {
        "mse": mse.detach(),
        "uncertainty": u_term.detach(),
        "safety": s_term.detach(),
        "total": total.detach(),
    }
    return total, parts


class EnsembleENAdapter(nn.Module):
    """Have an M-member ensemble present the single-EN interface the CPN epoch
    expects: __call__(en_in) -> mu_bar (batch, out_dim), so the existing
    `cur_pred = self.en(en_in)` line works unchanged. 

    After each call, the last per-member outputs are cached so the epoch can read the uncertainty for the
    loss via .last_uncertainty(). reset() forwards to the ensemble."""

    def __init__(self, ensemble: EnsembleLike,
                 estimator: Estimator = Estimator.MAX_FROBENIUS):
        super().__init__()
        self.ensemble = ensemble
        self.estimator = estimator
        self._last_input = None

    def forward(self, en_in: torch.Tensor) -> torch.Tensor:
        mu, _var = self.ensemble.forward(en_in)     # (M, batch, out_dim)
        self._last_input = en_in
        return mu.mean(dim=0)                        # mu_bar, (batch, out_dim)

    def last_uncertainty(self) -> torch.Tensor:
        """Uncertainty for the most recent forward input, (batch,)."""
        if self._last_input is None:
            raise RuntimeError("Call the adapter on an input before last_uncertainty()")
        return uncertainty(self.ensemble, self._last_input, self.estimator)

    def reset(self):
        self.ensemble.reset()
        self._last_input = None
