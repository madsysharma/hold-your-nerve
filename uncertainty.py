"""
uncertainty.py

The single uncertainty interface the trust-aware loss consumes and the estimator variants to compare and the calibration analysis (the H4 finding).

Everything is dispatched through one function, uncertainty(ensemble, x_t, estimator), so the rest of the system never depends on which estimator is used.

All four estimators and the calibration analysis are implemented. MAX_FROBENIUS, DISAGREEMENT and SINGLE_VAR are closed-form functions of the members' moments; MC_DROPOUT resamples.
"""
from enum import Enum
from typing import Tuple

import torch

from ensemble_types import EnsembleLike


#3 error axes the calibration analysis is run separately for (H4).
ERROR_AXES = ("parametric", "distributional", "structural")

DEFAULT_MC_PASSES = 16


class Estimator(str, Enum):
    MAX_FROBENIUS = "max_frobenius"   # MOPO: max_i ||Sigma_i||_F   (aleatoric-based)
    DISAGREEMENT = "disagreement"     # Variance across member means (epistemic)
    MC_DROPOUT = "mc_dropout"         # Predictive variance across stochastic passes
    SINGLE_VAR = "single_var"         # One member's predicted variance: pure-aleatoric baseline


def uncertainty(ensemble: EnsembleLike, x_t: torch.Tensor,
                estimator: Estimator = Estimator.MAX_FROBENIUS,
                *, n_passes: int = DEFAULT_MC_PASSES) -> torch.Tensor:
    """Batch-wise scalar uncertainty u(state, theta) for one timestep.

    Args:
        ensemble: any EnsembleLike (see ensemble_types.py).
        x_t: EN input for one timestep, (batch, in_dim).
        estimator: which uncertainty definition to use.
        n_passes: stochastic passes per member, MC_DROPOUT only.
    Returns:
        (batch,) tensor of non-negative uncertainties.
    """
    if estimator == Estimator.MC_DROPOUT:
        # Branches before the shared forward below: it runs its own passes, and
        # the extra call would advance recurrent members' hidden state.
        return _mc_dropout(ensemble, x_t, n_passes)

    mu, var = ensemble.forward(x_t)                     # (M, batch, out_dim)
    return _from_moments(mu, var, estimator)


def _from_moments(mu: torch.Tensor, var: torch.Tensor,
                  estimator: Estimator) -> torch.Tensor:
    """The estimators that are a closed-form function of the members' moments.

    Split out so the calibration analysis can score a batch from a single
    forward pass instead of calling forward once for u and again for the error.
    """
    if estimator == Estimator.MAX_FROBENIUS:
        # Each member is diagonal-Gaussian, so ||Sigma_i||_F = sqrt(sum_j var_ij^2).

        frob = torch.sqrt((var ** 2).sum(dim=-1))       # (M, batch)
        return frob.max(dim=0).values                   # (batch,)

    if estimator == Estimator.DISAGREEMENT:
        # Spread of member means, summed over output dims.

        return mu.var(dim=0, unbiased=False).sum(dim=-1)  # (batch,)

    if estimator == Estimator.SINGLE_VAR:
        # One member's own predicted variance, carrying no cross-member
        # information at all. The pure-aleatoric baseline: if it tracks
        # MAX_FROBENIUS, the ensemble is buying nothing over a single
        # probabilistic network.

        return var[0].sum(dim=-1)                       # (batch,)

    if estimator == Estimator.MC_DROPOUT:
        raise ValueError("MC_DROPOUT is not a function of the moments; call uncertainty().")

    raise ValueError(f"unknown estimator: {estimator}")


def _dropout_modules(ensemble: EnsembleLike) -> list:
    #every dropout layer we can reach, on the ensemble itself or on its members
    #match by class name so Dropout1d/2d/3d and AlphaDropout all count, whatever torch version
    found, seen = [], set()
    candidates = [ensemble]
    for attr in ("members", "ens", "nets"):
        got = getattr(ensemble, attr, None)
        if got is not None:
            candidates.extend(got)

    for obj in candidates:
        if not isinstance(obj, torch.nn.Module):
            continue
        for m in obj.modules():
            if "Dropout" in type(m).__name__ and id(m) not in seen:
                seen.add(id(m))
                found.append(m)
    return found


def _mc_dropout(ensemble: EnsembleLike, x_t: torch.Tensor,
                n_passes: int) -> torch.Tensor:
    #switch dropout ON at inference and run every member n_passes times
    #spread across those passes = that member's epistemic estimate, then average the members. (batch,)
    #careful: EnsembleLike has no way to save/restore hidden state, so n_passes calls advance a
    #recurrent member n_passes steps. Only safe at a step whose state you don't mind burning
    if n_passes < 2:
        raise ValueError(f"MC_DROPOUT needs n_passes >= 2 to have a variance, got {n_passes}.")

    mods = _dropout_modules(ensemble)
    if not mods:
        raise RuntimeError(
            "MC_DROPOUT found no dropout layers on this ensemble, so every pass "
            "would be identical and u would be exactly zero. Note that members "
            "built as LSTM(dropout=p) expose a float, not a module, and so have "
            "nothing to switch on at inference."
        )

    was_training = [m.training for m in mods]
    for m in mods:
        m.train()                                       # dropout ON at inference
    try:
        passes = [ensemble.forward(x_t)[0] for _ in range(n_passes)]
    finally:
        for m, prev in zip(mods, was_training):
            m.train(prev)                               # leave the ensemble as we found it

    stacked = torch.stack(passes, dim=0)                          # (P, M, batch, out_dim)
    per_member = stacked.var(dim=0, unbiased=False).sum(dim=-1)   # (M, batch)
    return per_member.mean(dim=0)                                 # (batch,)


def calibration_curve(ensemble: EnsembleLike, dataloader, error_axis: str,
                      *, estimator: Estimator = Estimator.MAX_FROBENIUS,
                      n_bins: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    #does u actually predict the EN's error? bin by u, report the real error in each bin
    #dataloader yields (x_t, target), either one step (batch, in_dim) or a rollout (batch, T, in_dim)
    #real error = |mu_bar - target| averaged over the 50 output dims
    #equal-COUNT bins, not equal-width: u is right-skewed so fixed widths dump nearly everything in bin 0
    #calibrated = the curve rises. H4 says it rises on parametric/distributional and goes flat on
    #structural, where members sharing a wrong architecture agree confidently while being wrong
    if error_axis not in ERROR_AXES:
        raise ValueError(f"unknown error axis: {error_axis!r}. Options: {ERROR_AXES}")
    if estimator == Estimator.MC_DROPOUT:
        raise ValueError(
            "calibration_curve scores u and the error from one forward pass; "
            "MC_DROPOUT resamples, so there is no single prediction to bin against."
        )

    us, errs = [], []
    for x_t, target in dataloader:
        for x, y in _walk_steps(x_t, target):
            mu, var = ensemble.forward(x)               # (M, batch, out_dim)
            us.append(_from_moments(mu, var, estimator).detach().flatten())
            errs.append((mu.mean(dim=0) - y).abs().mean(dim=-1).detach().flatten())

    if not us:
        raise ValueError("dataloader yielded no batches; nothing to calibrate.")

    u_all = torch.cat(us)
    e_all = torch.cat(errs)

    qs = torch.linspace(0.0, 1.0, n_bins + 1, device=u_all.device, dtype=u_all.dtype)
    edges = torch.quantile(u_all, qs)
    idx = torch.bucketize(u_all, edges[1:-1].contiguous())

    centers, means = [], []
    for b in range(n_bins):
        sel = idx == b
        if not bool(sel.any()):                         # ties collapse bins; skip the empties
            continue
        centers.append(u_all[sel].mean())
        means.append(e_all[sel].mean())

    return torch.stack(centers), torch.stack(means)


def _walk_steps(x_t: torch.Tensor, target: torch.Tensor):
    #hand back (x, y) one timestep at a time, for a single step or a whole rollout
    #members are recurrent, so a rollout gets walked in order instead of flattened into one big batch
    if x_t.dim() == 2:
        yield x_t, target
        return
    if x_t.dim() != 3:
        raise ValueError(f"expected (batch, in_dim) or (batch, T, in_dim), got {tuple(x_t.shape)}")
    if target.shape[1] != x_t.shape[1]:
        raise ValueError(
            f"x_t has {x_t.shape[1]} timesteps but target has {target.shape[1]}."
        )
    for t in range(x_t.shape[1]):
        yield x_t[:, t], target[:, t]
