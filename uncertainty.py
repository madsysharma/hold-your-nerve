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


def uncertainty_from(mu: torch.Tensor, var: torch.Tensor,
                     estimator: Estimator = Estimator.MAX_FROBENIUS) -> torch.Tensor:
    """Compute a batch-wise uncertainty from ALREADY-COMPUTED member outputs, so a
    caller that has just run the ensemble forward does not have to run it again.
    This is what lets EnsembleENAdapter read the uncertainty of the prediction it
    just made without a second forward (which would double-step recurrent members).

    mu, var: (M, batch, out_dim) per-member means and predicted variances.
    Returns (batch,). MC_DROPOUT is not available here because it inherently needs
    multiple stochastic passes; call uncertainty(ensemble, x_t, MC_DROPOUT) instead.
    """
    if estimator == Estimator.MAX_FROBENIUS:
        frob = torch.sqrt((var ** 2).sum(dim=-1))       # (M, batch)
        return frob.max(dim=0).values                   # (batch,)
    if estimator == Estimator.DISAGREEMENT:
        return mu.var(dim=0, unbiased=False).sum(dim=-1)  # (batch,)
    if estimator == Estimator.SINGLE_VAR:
        return var[0].sum(dim=-1)                       # (batch,)
    if estimator == Estimator.MC_DROPOUT:
        raise ValueError(
            "MC_DROPOUT needs multiple stochastic passes and cannot be computed "
            "from a single cached forward; call uncertainty(ensemble, x_t, "
            "Estimator.MC_DROPOUT) directly.")
    raise ValueError(f"unknown estimator: {estimator}")


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
    # MAX_FROBENIUS (diagonal-Gaussian ||Sigma_i||_F), DISAGREEMENT (spread of
    # member means), and SINGLE_VAR (one member's variance: the pure-aleatoric
    # baseline) are all pure functions of (mu, var); share that logic so a caller
    # holding (mu, var) can reuse it without a second forward.
    return uncertainty_from(mu, var, estimator)


def _dropout_modules(ensemble: EnsembleLike) -> list:
    #every dropout layer in the ensemble. .modules() already walks into the members
    #match by class name so Dropout1d/2d/3d and AlphaDropout all count, whatever torch version
    if not isinstance(ensemble, torch.nn.Module):
        return []
    return [m for m in ensemble.modules() if "Dropout" in type(m).__name__]


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
    passes = [ensemble.forward(x_t)[0] for _ in range(n_passes)]
    for m, prev in zip(mods, was_training):
        m.train(prev)                                   # leave the ensemble as we found it

    stacked = torch.stack(passes, dim=0)                          # (P, M, batch, out_dim)
    per_member = stacked.var(dim=0, unbiased=False).sum(dim=-1)   # (M, batch)
    return per_member.mean(dim=0)                                 # (batch,)


def calibration_curve(ensemble: EnsembleLike, dataloader, error_axis: str,
                      *, estimator: Estimator = Estimator.MAX_FROBENIUS,
                      n_bins: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    #does u actually predict the EN's error? bin by u, report the real error in each bin
    #dataloader yields (x_t, target) one timestep at a time: (batch, in_dim) and (batch, out_dim)
    #real error = |mu_bar - target| averaged over the 50 output dims
    #calibrated = the curve rises. H4 says it rises on parametric/distributional and goes flat on
    #structural, where members sharing a wrong architecture agree confidently while being wrong
    if error_axis not in ERROR_AXES:
        raise ValueError(f"unknown error axis: {error_axis!r}. Options: {ERROR_AXES}")

    us, errs = [], []
    for x, y in dataloader:
        us.append(uncertainty(ensemble, x, estimator).detach().flatten())
        mu, _var = ensemble.forward(x)                  # (M, batch, out_dim)
        errs.append((mu.mean(dim=0) - y).abs().mean(dim=-1).detach().flatten())

    if not us:
        raise ValueError("dataloader yielded no batches; nothing to calibrate.")

    u_all = torch.cat(us)
    e_all = torch.cat(errs)

    edges = torch.linspace(u_all.min().item(), u_all.max().item(), n_bins + 1,
                           device=u_all.device, dtype=u_all.dtype)
    idx = torch.bucketize(u_all, edges[1:-1].contiguous())

    centers, means = [], []
    for b in range(n_bins):
        sel = idx == b
        if not bool(sel.any()):                         # skip bins nothing landed in
            continue
        centers.append(u_all[sel].mean())
        means.append(e_all[sel].mean())

    return torch.stack(centers), torch.stack(means)


def plot_reliability_diagrams(predicted_uncertainties: dict, actual_errors: dict,
                              output_path: str = "fig_reliability_diagrams.png"):
    """Render reliability diagrams of predicted uncertainty vs actual EN error,
    one panel per error axis. This is the plotting companion to calibration_curve
    (which produces the binned points); pass it per-axis arrays of predicted
    uncertainty and the matching true residual errors.

    predicted_uncertainties: {axis_name: array of predicted uncertainties}
    actual_errors:           {axis_name: array of true residual errors}
    output_path:             where to save the figure.

    matplotlib and numpy are imported lazily so importing this module for the
    estimators alone does not require them.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    axes_labels = list(predicted_uncertainties.keys())
    fig, axes = plt.subplots(1, len(axes_labels),
                             figsize=(5 * len(axes_labels), 4), sharey=True)
    if len(axes_labels) == 1:
        axes = [axes]

    for ax, axis_name in zip(axes, axes_labels):
        unc = np.array(predicted_uncertainties[axis_name])
        err = np.array(actual_errors[axis_name])

        num_bins = 8
        bins = np.linspace(np.min(unc), np.max(unc), num_bins + 1)
        bin_indices = np.digitize(unc, bins)

        bin_unc_means, bin_err_means = [], []
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if np.any(mask):
                bin_unc_means.append(np.mean(unc[mask]))
                bin_err_means.append(np.mean(err[mask]))

        ax.plot(bin_unc_means, bin_err_means, "o-", linewidth=2,
                color="crimson", label="Observed error")
        lo = min(np.min(unc), np.min(err))
        hi = max(np.max(unc), np.max(err))
        ax.plot([lo, hi], [lo, hi], "--", color="gray", label="Ideal calibration")
        ax.set_title(f"Error axis: {axis_name.capitalize()}")
        ax.set_xlabel("Predicted uncertainty")
        ax.set_ylabel("Actual EN error")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved calibration reliability diagrams to {output_path}")
