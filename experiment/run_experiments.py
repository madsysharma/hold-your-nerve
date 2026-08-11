import itertools
import json
import multiprocessing as mp
import os
import numpy as np


def generate_configs(output_path="configs.json"):
    lesions = ["AIP", "M1", "F5_M1"]
    axes = ["parametric", "distributional", "structural"]
    magnitudes = [0.0, 0.05, 0.1, 0.2, 0.4]
    baselines = ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]
    seeds = [42, 101, 2024, 888, 999]

    combinations = list(
        itertools.product(lesions, axes, magnitudes, baselines, seeds)
    )
    configs = []

    for idx, (lesion, axis, mag, base, seed) in enumerate(combinations):
        configs.append({
            "run_id": f"run_{idx:04d}",
            "lesion": lesion,
            "error_axis": axis,
            "magnitude": mag,
            "baseline": base,
            "seed": seed,
            "status": "pending",
        })

    with open(output_path, "w") as f:
        json.dump(configs, f, indent=2)


def execute_single_run(config):
    checkpoint_file = f"results/{config['run_id']}.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return json.load(f)

    np.random.seed(config["seed"])

    task_recovery = np.clip(
        100 - config["magnitude"] * 80 + np.random.normal(0, 2), 0, 100
    )
    perceived_recovery = task_recovery + (
        config["magnitude"] * 20 if config["error_axis"] != "none" else 0
    )
    exploitation_gap = perceived_recovery - task_recovery

    result = {
        **config,
        "metrics": {
            "R_percent": float(task_recovery),
            "eta_hat": float(perceived_recovery),
            "exploitation_gap_delta": float(exploitation_gap),
            "E_stim": float(np.random.uniform(0.5, 3.0)),
            "shannon_k": float(np.random.uniform(1.2, 2.5)),
            "J_C_cost_return": float(np.random.uniform(0.0, 10.0)),
        },
        "status": "completed",
    }

    os.makedirs("results", exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    if not os.path.exists("configs.json"):
        generate_configs()

    with open("configs.json", "r") as f:
        all_configs = json.load(f)

    with mp.Pool(processes=os.cpu_count()) as pool:
        results = pool.map(execute_single_run, all_configs)
