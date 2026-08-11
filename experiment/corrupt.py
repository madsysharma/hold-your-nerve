import copy
import numpy as np
import torch
import torch.nn as nn


def corrupt(EN_model, axis: str, magnitude: float):
    """Unified corruption interface for Emulator Networks (EN).

    Args:
        EN_model (nn.Module): PyTorch Emulator Network model.
        axis (str): One of 'parametric', 'distributional', or 'structural'.
        magnitude (float): Noise magnitude (sigma_p, delta angle/shift, or
          variant ID).

    Returns:
        nn.Module: Corrupted emulator network instance.
    """
    corrupted_model = copy.deepcopy(EN_model)

    if axis == "parametric":
        # w_new = w + epsilon, where epsilon ~ N(0, sigma_p^2 * Var(w))
        sigma_p = magnitude
        with torch.no_grad():
            for param in corrupted_model.parameters():
                if param.requires_grad:
                    var_w = torch.var(param)
                    if var_w == 0 or torch.isnan(var_w):
                        continue
                    std_noise = sigma_p * torch.sqrt(var_w)
                    noise = torch.randn_like(param) * std_noise
                    param.add_(noise)

    elif axis == "distributional":
        # Applies planar rotation R(delta) and translation T(delta) to latent features
        delta = magnitude

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                latent = output[0]
            else:
                latent = output

            # Construct 2D rotation matrix R(delta)
            cos_d, sin_d = np.cos(delta), np.sin(delta)
            R = torch.tensor(
                [[cos_d, -sin_d], [sin_d, cos_d]],
                dtype=latent.dtype,
                device=latent.device,
            )

            # Rotate first two dimensions and translate across features
            transformed = latent.clone()
            if transformed.shape[-1] >= 2:
                transformed[..., :2] = torch.matmul(transformed[..., :2], R)
            transformed = transformed + delta
            return (
                (transformed, output[1])
                if isinstance(output, tuple)
                else transformed
            )

        # Attach forward hook to latent/recurrent layer
        target_layer = getattr(
            corrupted_model,
            "rnn",
            getattr(corrupted_model, "fc1", corrupted_model),
        )
        target_layer.register_forward_hook(hook_fn)

    elif axis == "structural":
        # Structural variants: 0 -> 20-unit LSTM, 1 -> GRU, 2 -> Electrode spread mismatch
        variant_id = int(magnitude)
        corrupted_model = _build_structural_variant(
            variant_id, base_model=EN_model
        )

    return corrupted_model


def _build_structural_variant(variant_id: int, base_model):
    """Instantiates architectural mismatched surrogates."""
    if variant_id == 0:
        # 20-unit LSTM bottleneck mismatch
        return nn.LSTM(
            input_size=base_model.input_size
            if hasattr(base_model, "input_size")
            else 100,
            hidden_size=20,
            batch_first=True,
        )
    elif variant_id == 1:
        # GRU architectural mismatch
        return nn.GRU(
            input_size=base_model.input_size
            if hasattr(base_model, "input_size")
            else 100,
            hidden_size=base_model.hidden_size
            if hasattr(base_model, "hidden_size")
            else 64,
            batch_first=True,
        )
    elif variant_id == 2:
        # Spatial electrode dispersion mismatch
        mismatched_model = copy.deepcopy(base_model)
        if hasattr(mismatched_model, "sigma_spread"):
            mismatched_model.sigma_spread *= 2.5
        return mismatched_model
    else:
        raise ValueError(f"Unknown structural variant ID: {variant_id}")
