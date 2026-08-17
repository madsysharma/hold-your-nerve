import glob
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu


def load_results(results_dir="results"):
    """Loads all finished run JSON files from the results directory."""
    results = []
    for filepath in glob.glob(f"{results_dir}/run_*.json"):
        with open(filepath, "r") as f:
            results.append(json.load(f))
    return results


def plot_figure_1_exploitation_gap(results, output_path="fig1_exploitation_gap.png"):
    """Figure 1: Plot Exploitation Gap (Delta) vs. Error Magnitude across error axes."""
    magnitudes = [0.0, 0.05, 0.1, 0.2, 0.4]
    axes = ["parametric", "distributional", "structural"]

    plt.figure(figsize=(8, 5))

    for axis in axes:
        gaps_mean = []
        gaps_std = []
        for mag in magnitudes:
            filtered_vals = [
                r["metrics"]["exploitation_gap_delta"]
                for r in results
                if r.get("error_axis") == axis and r.get("magnitude") == mag
            ]
            if filtered_vals:
                gaps_mean.append(np.mean(filtered_vals))
                gaps_std.append(np.std(filtered_vals))
            else:
                gaps_mean.append(0.0)
                gaps_std.append(0.0)

        plt.errorbar(
            magnitudes,
            gaps_mean,
            yerr=gaps_std,
            marker="o",
            capsize=4,
            label=f"{axis.capitalize()} Error",
        )

    plt.xlabel("Error Magnitude")
    plt.ylabel("Exploitation Gap (Δ)")
    plt.title("Figure 1: Exploitation Gap vs. Error Magnitude")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Figure 1 to {output_path}")


def compute_holm_bonferroni_mann_whitney(group_a, group_b, alpha=0.05):
    """Performs Mann-Whitney U test with Holm-Bonferroni correction across multiple baseline metrics."""
    p_values = []
    keys = list(group_a.keys())

    for key in keys:
        _, p_val = mannwhitneyu(group_a[key], group_b[key], alternative="two-sided")
        p_values.append(p_val)

    # Holm-Bonferroni step-down correction
    sorted_indices = np.argsort(p_values)
    m = len(p_values)
    results = {}

    for rank, idx in enumerate(sorted_indices):
        adjusted_alpha = alpha / (m - rank)
        key = keys[idx]
        results[key] = {
            "p_value": float(p_values[idx]),
            "significant": bool(p_values[idx] <= adjusted_alpha),
            "adjusted_alpha": float(adjusted_alpha),
        }

    return results


if __name__ == "__main__":
    data = load_results()
    if data:
        plot_figure_1_exploitation_gap(data)
        print(f"Loaded {len(data)} completed run records for analysis.")
    else:
        print("No result files found in results/. Run run_experiments.py first.")
