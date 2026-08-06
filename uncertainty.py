"""
uncertainty.py

The single uncertainty interface the trust-aware loss consumes and the estimator variants to compare and the calibration analysis (the H4 finding).

Everything is dispatched through one function, uncertainty(ensemble, x_t, estimator), so the rest of the system never depends on which estimator is used.

Two estimators are implemented as reference; the other two and the calibration analysis are to be filled in by the collaborator responsible.
"""
from enum import Enum

import torch

from ensemble_types import EnsembleLike


class Estimator(str, Enum):
    MAX_FROBENIUS = "max_frobenius"   # MOPO: max_i ||Sigma_i||_F   (aleatoric-based)
    DISAGREEMENT = "disagreement"     # Variance across member means (epistemic)
    MC_DROPOUT = "mc_dropout"         # TODO(by )
    SINGLE_VAR = "single_var"         # One member's predicted variance: TODO (by )


def uncertainty(ensemble: EnsembleLike, x_t: torch.Tensor,
                estimator: Estimator = Estimator.MAX_FROBENIUS) -> torch.Tensor:
    """Batch-wise scalar uncertainty u(state, theta) for one timestep.

    Args:
        ensemble: any EnsembleLike (see ensemble_types.py).
        x_t: EN input for one timestep, (batch, in_dim).
        estimator: which uncertainty definition to use.
    Returns:
        (batch,) tensor of non-negative uncertainties.
    """
    mu, var = ensemble.forward(x_t)                     # (M, batch, out_dim)

    if estimator == Estimator.MAX_FROBENIUS:
        # Each member is diagonal-Gaussian, so ||Sigma_i||_F = sqrt(sum_j var_ij^2).

        frob = torch.sqrt((var ** 2).sum(dim=-1))       # (M, batch)
        return frob.max(dim=0).values                   # (batch,)

    if estimator == Estimator.DISAGREEMENT:
        # Spread of member means, summed over output dims.

        return mu.var(dim=0, unbiased=False).sum(dim=-1)  # (batch,)

    if estimator == Estimator.MC_DROPOUT:
        raise NotImplementedError(
            "TODO(by ): enable dropout at inference, run T stochastic passes per "
            "member, and return the predictive variance across passes.")

    if estimator == Estimator.SINGLE_VAR:
        raise NotImplementedError(
            "TODO(by ): return a single member's predicted variance (var[0]) summed "
            "over output dims - the pure-aleatoric baseline for the ablation.")

    raise ValueError(f"unknown estimator: {estimator}")


def calibration_curve(ensemble: EnsembleLike, dataloader, error_axis: str):
    """Reliability data: predicted uncertainty vs actual EN error, per error axis.

    TODO(by ): for each batch, compute u(ensemble, x_t) and the true EN error |mu_bar - target|, bin by predicted uncertainty, 
    and return (bin_centers, mean_actual_error) so we can plot a reliability diagram.

    Run it separately for error_axis in {"parametric", "distributional", "structural"}: the H4 prediction is that calibration looks good on the first two 
    and fails on structural (members sharing a wrong architecture agree confidently).
    """
    raise NotImplementedError("TODO(by ): calibration / reliability analysis (H4).")
