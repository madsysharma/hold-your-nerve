"""
safety.py

Defines the safety cost C_safety(theta) and the two caution styles the trust-aware loss can use. 

The pieces that are fully determined (stimulation energy; the soft and hard caution functions) are already implemented:
the physical damage proxy and the out-of-distribution fraction are to be filled in by .
"""
from dataclasses import dataclass
import torch
from ensemble_types import EnsembleLike


@dataclass
class SafetyWeights:
    """Relative weights for clubbing the safety terms into one cost."""
    energy: float = 1.0
    charge: float = 0.0     # Set > 0 once charge_damage_fraction is implemented
    ood: float = 0.0        # Set > 0 once ood_fraction is implemented


def stim_energy(theta: torch.Tensor) -> torch.Tensor:
    """Stimulation energy per batch element: sum of squared stimulation.
    theta: (batch, n_channels) for one step, or (batch, T, n_channels).
    Returns (batch,)."""
    dims = tuple(range(1, theta.dim()))
    return (theta ** 2).sum(dim=dims)


def charge_damage_fraction(theta: torch.Tensor, k_threshold: float = 1.85) -> torch.Tensor:
    """Fraction of timesteps exceeding the Shannon damage criterion.

    TODO(by ): map theta to charge-per-phase Q and charge-density D, compute
    k = log10(D) + log10(Q), and return the fraction of steps with k >= k_threshold
    (default 1.85). 

    Add the Cogan ~4 nC/phase microelectrode variant as an option.

    Return (batch,)."""
    raise NotImplementedError("TODO(by ): Shannon k = log10(D)+log10(Q) >= 1.85 proxy.")


def ood_fraction(ensemble: EnsembleLike, x_t: torch.Tensor,
                 width_threshold: float) -> torch.Tensor:
    """Fraction of steps where the model is out of its trusted region.

    TODO(by ): flag steps whose (state, theta) fall outside the EN's training
    support, eg: where a conformal interval width (or the ensemble uncertainty)
    exceeds width_threshold. 

    Return (batch,)."""
    raise NotImplementedError("TODO(D): OOD / conformal-width fraction.")


def safety_cost(theta: torch.Tensor, *, ensemble: EnsembleLike = None,
                x_t: torch.Tensor = None, weights: SafetyWeights = None) -> torch.Tensor:
    """Combined safety cost C_safety, (batch,). 

    Only the energy term is active by default; the charge and OOD terms switch on when their weight is > 0 and their
    implementation lands."""
    weights = weights or SafetyWeights()
    cost = weights.energy * stim_energy(theta)
    if weights.charge > 0.0:
        cost = cost + weights.charge * charge_damage_fraction(theta)
    if weights.ood > 0.0:
        cost = cost + weights.ood * ood_fraction(ensemble, x_t, width_threshold=1.0)
    return cost


# ---- Caution styles (how uncertainty turns into a penalty / action) --------
def soft_penalty(u: torch.Tensor, lam: float) -> torch.Tensor:
    """MOPO-style graded penalty: lam * u. 
    Returns same shape as u."""
    return lam * u


def hard_halt(u: torch.Tensor, threshold: float, penalty: float) -> torch.Tensor:
    """MOReL-style hard cutoff: a large penalty wherever u exceeds threshold, zero elsewhere.
    Returns same shape as u. (The paired action - clamping theta
    to zero on HALT - is applied at the call site in the training loop.)"""
    return torch.where(u > threshold,
                       torch.full_like(u, penalty),
                       torch.zeros_like(u))
