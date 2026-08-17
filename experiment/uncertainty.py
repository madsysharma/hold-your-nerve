import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class UncertaintyEstimator:
    """Implements MOPO penalty, MC-Dropout, and Single-Network heteroscedastic variance estimators."""

    @staticmethod
    def mc_dropout_variance(model: nn.Module, x: torch.Tensor, num_samples: int = 20) -> torch.Tensor:
        """Calculates epistemic variance via Monte Carlo Dropout across forward passes.
        
        Forces dropout layers into train mode during inference.
        """
        model.eval()
        # Enable dropout layers specifically during evaluation
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                preds = model(x)
                if isinstance(preds, tuple):
                    preds = preds[0]
                predictions.append(preds.unsqueeze(0))

        preds_stacked = torch.cat(predictions, dim=0)
        return torch.var(preds_stacked, dim=0)

    @staticmethod
    def mopo_penalty(ensemble_models: list, x: torch.Tensor, lambda_penalty: float = 1.0) -> torch.Tensor:
        """Model-Based Offline Policy Optimization (MOPO) uncertainty penalty.
        
        Computes maximum variance/std across ensemble predictions scaled by lambda.
        """
        preds = []
        with torch.no_grad():
            for m in ensemble_models:
                m.eval()
                out = m(x)
                if isinstance(out, tuple):
                    out = out[0]
                preds.append(out.unsqueeze(0))

        preds_stacked = torch.cat(preds, dim=0)
        ensemble_var = torch.var(preds_stacked, dim=0)
        max_std = torch.sqrt(torch.max(ensemble_var, dim=-1)[0])
        return lambda_penalty * max_std

    @staticmethod
    def single_network_variance(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Extracts aleatoric log-variance predicted by a single heteroscedastic head network."""
        model.eval()
        with torch.no_grad():
            out = model(x)
            if isinstance(out, tuple) and len(out) >= 2:
                _, log_var = out[0], out[1]
                return torch.exp(log_var)
            raise ValueError("Model output must be a tuple containing (mean, log_variance).")


def plot_reliability_diagrams(
    predicted_uncertainties: dict,
    actual_errors: dict,
    output_path: str = "fig_reliability_diagrams.png"
):
    """Generates reliability diagrams of predicted uncertainty vs. actual EN error split by error axis.
    
    Args:
        predicted_uncertainties (dict): {axis_name: np.ndarray of predicted uncertainties}
        actual_errors (dict): {axis_name: np.ndarray of true residual errors}
        output_path (str): Filepath to save the generated figure.
    """
    axes_labels = list(predicted_uncertainties.keys())
    fig, axes = plt.subplots(1, len(axes_labels), figsize=(5 * len(axes_labels), 4), sharey=True)
    if len(axes_labels) == 1:
        axes = [axes]

    for ax, axis_name in zip(axes, axes_labels):
        unc = np.array(predicted_uncertainties[axis_name])
        err = np.array(actual_errors[axis_name])

        # Bin predicted uncertainty into equal-width quantiles
        num_bins = 8
        bins = np.linspace(np.min(unc), np.max(unc), num_bins + 1)
        bin_indices = np.digitize(unc, bins)

        bin_unc_means = []
        bin_err_means = []

        for i in range(1, len(bins)):
            mask = bin_indices == i
            if np.any(mask):
                bin_unc_means.append(np.mean(unc[mask]))
                bin_err_means.append(np.mean(err[mask]))

        ax.plot(bin_unc_means, bin_err_means, "o-", linewidth=2, color="crimson", label="Observed Error")
        
        # Perfect calibration reference line
        min_val = min(np.min(unc), np.min(err))
        max_val = max(np.max(unc), np.max(err))
        ax.plot([min_val, max_val], [min_val, max_val], "--", color="gray", label="Ideal Calibration")

        ax.set_title(f"Error Axis: {axis_name.capitalize()}")
        ax.set_xlabel("Predicted Uncertainty")
        ax.set_ylabel("Actual EN Error")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved calibration reliability diagrams to {output_path}")
