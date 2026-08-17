# experiment/corrupt.py

import copy
import torch
import torch.nn as nn

# Adjust import according to your repo structure (e.g., from stage1.error_injection import ErrorInjector)
try:
    from error_injection import ErrorInjector
except ImportError:
    from stage1.error_injection import ErrorInjector


def corrupt(EN_model: nn.Module, axis: str, magnitude: float) -> nn.Module:
    """Unified corruption interface for Emulator Networks (EN).

    Delegates transformation logic directly to Stage 1 ErrorInjector.

    Args:
        EN_model (nn.Module): PyTorch Emulator Network model.
        axis (str): One of 'parametric', 'distributional', or 'structural'.
        magnitude (float): Noise magnitude (sigma_p, delta angle/shift, or variant ID).

    Returns:
        nn.Module: Corrupted emulator network instance.
    """
    corrupted_model = copy.deepcopy(EN_model)

    if axis == "parametric":
        return ErrorInjector.apply_param_error(
            corrupted_model, sigma_p=float(magnitude)
        )

    elif axis == "distributional":
        delta_val = float(magnitude)
        # If model is an mRNN with num_neurons attribute, use hook drift
        if hasattr(corrupted_model, "num_neurons"):
            return ErrorInjector.apply_dist_error(corrupted_model, delta=delta_val)
        else:
            # Fallback for standard PyTorch EN modules using standalone tensor drift
            def hook_fn(module, input, output):
                latent = output[0] if isinstance(output, tuple) else output
                drifted = ErrorInjector.apply_distributional_error(
                    latent, delta=delta_val
                )
                return (drifted, output[1]) if isinstance(output, tuple) else drifted

            target_layer = getattr(
                corrupted_model,
                "rnn",
                getattr(corrupted_model, "fc1", corrupted_model),
            )
            target_layer.register_forward_hook(hook_fn)
            return corrupted_model

    elif axis == "structural":
        variant_id = int(magnitude)
        return _build_structural_variant(variant_id, base_model=EN_model)

    else:
        raise ValueError(f"Unknown corruption axis: {axis}")


def _build_structural_variant(variant_id: int, base_model: nn.Module) -> nn.Module:
    """Instantiates architectural mismatched surrogates using Stage 1 ErrorInjector."""
    in_dim = getattr(base_model, "input_size", getattr(base_model, "in_dim", 100))
    out_dim = getattr(base_model, "output_size", getattr(base_model, "out_dim", 10))

    if variant_id == 0:
        # 20-unit LSTM bottleneck mismatch
        return ErrorInjector.build_structural_en(
            in_dim=in_dim, out_dim=out_dim, num_neurons=20, cell_type="lstm"
        )
    elif variant_id == 1:
        # GRU architectural mismatch
        hidden_size = getattr(
            base_model, "hidden_size", getattr(base_model, "num_neurons", 64)
        )
        return ErrorInjector.build_structural_en(
            in_dim=in_dim, out_dim=out_dim, num_neurons=hidden_size, cell_type="gru"
        )
    elif variant_id == 2:
        # Spatial electrode dispersion mismatch
        mismatched_model = copy.deepcopy(base_model)
        if hasattr(mismatched_model, "sigma_spread"):
            mismatched_model.sigma_spread *= 2.5
        return mismatched_model
    else:
        raise ValueError(f"Unknown structural variant ID: {variant_id}")
