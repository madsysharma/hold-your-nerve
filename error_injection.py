import torch
import torch.nn as nn
import stim_model  # Defines EN architecture (used by build_structural_en)


class _DistributionalDrift:
    """
    Low-dimensional affine drift of the mRNN state: x -> R x + b.
    Built once from a seed (fixed manifold). R is a proper rotation (orthogonal for every delta, identity at delta=0), whose angle scales with delta, acting
    only inside a random k-dim subspace; b is a translation confined to that same subspace.

    This is the corrected drift (the earlier (1-d)*I + d*Q blend was not orthogonal and not low-dimensional). 
    Used by both the standalone tensor helper apply_distributional_error and the state-hook apply_dist_error.
    """
    def __init__(self, num_neurons, delta, k=8, seed=42):
        self.num_neurons = num_neurons
        self.delta = float(delta)
        if self.delta == 0.0:
            self.R = None
            self.b = None
            return
        k = max(1, min(int(k), num_neurons))
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        U, _ = torch.linalg.qr(torch.randn(num_neurons, k, generator=g))
        Kk = torch.randn(k, k, generator=g)
        Sk = Kk - Kk.T
        Sk = Sk / (Sk.norm() + 1e-12)
        A = U @ Sk @ U.T
        self.R = torch.linalg.matrix_exp(self.delta * torch.pi * A)
        self.b = self.delta * (U @ torch.randn(k, generator=g))

    def apply(self, x):
        if self.R is None:
            return x
        R = self.R.to(device=x.device, dtype=x.dtype)
        b = self.b.to(device=x.device, dtype=x.dtype)
        return x @ R.T + (b if x.dim() == 1 else b.unsqueeze(0))


class ErrorInjector:
    """Defines the unified interface for injecting errors into the EN."""

    # ---- Structural axis ---------------------------------------------------
    @staticmethod
    def build_structural_en(
        in_dim, out_dim, num_neurons=None, cell_type="lstm",
        activation=torch.nn.Tanh, cuda=None,
    ):
        """
        Structural error: builds the EN with mismatched num_neurons
        and/or cell type (lstm/gru/rnn).
        """
        return stim_model.get_stim_model(
            in_dim, out_dim, num_neurons=num_neurons,
            activation=activation, cuda=cuda, cell_type=cell_type,
        )

    @staticmethod
    def apply_observation_mismatch(mike, obs_model, module_idx, drop_neuron_mask=None):
        """
        Structural error: gives the EN an observer for a mike module built with a
        different sigma value.
        """
        return mike.observe_module(
            obs_model, module_idx, drop_neuron_mask=drop_neuron_mask
        )

    # ---- Parametric axis ---------------------------------------------------
    @staticmethod
    def apply_param_error(model: nn.Module, sigma_p: float) -> nn.Module:
        """
        Distorts model parameters: w <- w + eps,
        where eps ~ N(0, sigma_p^2 * Var(w)).
        Note: Std(eps) = sigma_p * Std(w).
        """
        if sigma_p <= 0.0:
            return model
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad or param.numel() > 1:
                    w_std = torch.std(param)
                    if torch.isnan(w_std) or w_std == 0:
                        continue
                    noise = torch.randn_like(param) * (sigma_p * w_std)
                    param.add_(noise)
        return model

    # ---- Distributional axis ----------------------------------------------
    @staticmethod
    def apply_distributional_error(hidden_state: torch.Tensor, delta: float,
                                   seed: int = 42, k: int = 8) -> torch.Tensor:
        """
        Standalone tensor form of the distributional drift: x -> R x + b, a low-dimensional rotation and translation. 
        Returns a drifted copy, same shape. 
        delta == 0 returns the input unchanged. 
        Row-vector convention: R x is x @ R.T. 
        (The training loop uses the hook form, apply_dist_error; this helper is kept for direct use and testing.)
        """
        if delta == 0.0:
            return hidden_state
        dim = hidden_state.shape[-1]
        drift = _DistributionalDrift(dim, delta, k=k, seed=seed)
        return drift.apply(hidden_state)

    @staticmethod
    def apply_dist_error(model: nn.Module, delta: float, k: int = 8,
                         seed: int = 42, mode: str = "onset") -> nn.Module:
        """
        State-hook distributional error: drift the brain's internal state x -> R x + b during rollout, so an emulator trained before the drift is
        evaluated after it. 

        Installs a forward hook on `model` (the mRNN) and returns it.

        model must expose .x, .prev_output, .activation_func, .num_neurons, .reset (which the Michaels mRNN does). 
        Passing an emulator raises ValueError, since only the brain has the state manifold that drifts.

        mode:
          "onset":      drift applied ONCE per trial, at the first step after each reset(). Non-compounding; the benchmark default.
          "persistent": drift applied EVERY step (a fixed R compounds to R^t over a trial; use small delta / short trials).

        Reusing on the same model is safe: any prior drift is removed first, so sweeping delta doesn't stack hooks. Use remove_dist_error to undo.
        """
        for attr in ("x", "prev_output", "activation_func", "num_neurons", "reset"):
            if not hasattr(model, attr):
                raise ValueError(
                    "apply_dist_error expects the brain model (MichaelsRNN) with a "
                    f"'{attr}' attribute; distributional drift acts on the brain "
                    "state, not the emulator. Pass the mRNN, not the EN.")
        if mode not in ("onset", "persistent"):
            raise ValueError("mode must be 'onset' or 'persistent'")

        ErrorInjector.remove_dist_error(model)
        if delta == 0.0:
            return model

        drift = _DistributionalDrift(model.num_neurons, delta, k=k, seed=seed)
        model._drift_pending = (mode == "onset")

        if mode == "onset":
            original_reset = model.reset

            def reset_with_rearm(*args, _orig=original_reset, _m=model, **kwargs):
                out = _orig(*args, **kwargs)
                _m._drift_pending = True
                return out

            model._drift_original_reset = original_reset
            model.reset = reset_with_rearm

        def _hook(module, _inputs, _output):
            if module.x is None:
                return
            if mode == "onset":
                if not getattr(module, "_drift_pending", False):
                    return
                module._drift_pending = False
            module.x = drift.apply(module.x)
            module.prev_output = module.activation_func(module.x)

        model._drift_handle = model.register_forward_hook(_hook)
        return model

    @staticmethod
    def remove_dist_error(model: nn.Module) -> nn.Module:
        """Remove a drift hook installed by apply_dist_error and restore reset()."""
        handle = getattr(model, "_drift_handle", None)
        if handle is not None:
            handle.remove()
            model._drift_handle = None
        orig = getattr(model, "_drift_original_reset", None)
        if orig is not None:
            model.reset = orig
            model._drift_original_reset = None
        if hasattr(model, "_drift_pending"):
            model._drift_pending = False
        return model
