import torch
import torch.nn as nn
import stim_model  # Defines the EN architecture (used by build_structural_en)


# ---------------------------------------------------------------------------
# Low-dimensional affine drift of the mRNN state: x -> R x + b.
# Built once from a seed (fixed manifold). R is a proper rotation (orthogonal for every delta, identity at delta=0), 
# whose angle scales with delta, acting only inside a random k-dim subspace; b is a translation confined to that same subspace. 
# This is the corrected drift (the earlier (1-d)*I + d*Q blend was not
# orthogonal and not low-dimensional).
# ---------------------------------------------------------------------------
class _DistributionalDrift:
    def __init__(self, num_neurons, delta, k=8, seed=42):
        self.num_neurons = num_neurons
        self.delta = float(delta)
        if self.delta == 0.0:
            self.R = None
            self.b = None
            self._A = None
            self._b_dir = None
            return
        k = max(1, min(int(k), num_neurons))
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        U, _ = torch.linalg.qr(torch.randn(num_neurons, k, generator=g))
        Kk = torch.randn(k, k, generator=g)
        Sk = Kk - Kk.T
        Sk = Sk / (Sk.norm() + 1e-12)
        self._A = U @ Sk @ U.T                       # Skew generator, rank <= k
        self._b_dir = U @ torch.randn(k, generator=g)  # Translation direction
        # Full drift (magnitude delta): used by onset / persistent modes.
        self.R = torch.linalg.matrix_exp(self.delta * torch.pi * self._A)
        self.b = self.delta * self._b_dir

    def apply(self, x):
        """Apply the FULL drift x -> R x + b (onset / persistent)."""
        if self.R is None:
            return x
        R = self.R.to(device=x.device, dtype=x.dtype)
        b = self.b.to(device=x.device, dtype=x.dtype)
        return x @ R.T + (b if x.dim() == 1 else b.unsqueeze(0))

    def increment(self, n):
        """Return the per-step increment (R_step, b_step) such that applying it
        n times accumulates to the full drift: R_step^n = R, n*b_step = b. 
        Used by the slow ramp: each step nudges the state by 1/n of the drift, so it
        wanders gradually to magnitude delta over n steps (one trial)."""
        if self._A is None:
            return None, None
        n = max(1, int(n))
        R_step = torch.linalg.matrix_exp((self.delta * torch.pi / n) * self._A)
        b_step = (self.delta / n) * self._b_dir
        return R_step, b_step

    def transform_at(self, frac):
        """Return (R, b) for the drift at a fraction of full magnitude, i.e.
        effective delta = frac*delta. 
        Used by multi-session drift, where the session-level drift is applied once per trial at a magnitude that ramps
        across trials. frac is clamped to [0, 1]."""
        if self._A is None:
            return None, None
        frac = max(0.0, min(1.0, float(frac)))
        R = torch.linalg.matrix_exp((frac * self.delta * torch.pi) * self._A)
        b = (frac * self.delta) * self._b_dir
        return R, b


class ErrorInjector:
    """Unified interface for injecting EN errors."""

    # ---- Structural axis ---------------------------------------------------
    @staticmethod
    def build_structural_en(
        in_dim, out_dim, num_neurons=None, cell_type="lstm",
        activation=torch.nn.Tanh, cuda=None,
    ):
        """
        Structural error: builds the EN with mismatched num_neurons and/or cell type (lstm/gru/rnn).
        """
        return stim_model.get_stim_model(
            in_dim, out_dim, num_neurons=num_neurons,
            activation=activation, cuda=cuda, cell_type=cell_type,
        )

    @staticmethod
    def apply_observation_mismatch(mike, obs_model, module_idx, drop_neuron_mask=None):
        """
        Structural error: gives the EN an observer for a mike module built with a different sigma value.
        """
        return mike.observe_module(
            obs_model, module_idx, drop_neuron_mask=drop_neuron_mask
        )

    # ---- Parametric axis ---------------------------------------------------
    @staticmethod
    def apply_param_error(model: nn.Module, sigma_p: float) -> nn.Module:
        """
        Distorts model parameters: w <- w + eps, where eps ~ N(0, sigma_p^2 * Var(w)).
        NOTE: Std(eps) = sigma_p * Std(w).
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
        Returns a drifted copy, same shape. delta == 0 returns the input unchanged. 
        Row-vector convention: R x is x @ R.T. (The training loop uses the hook form, apply_dist_error;
        this helper is kept for direct use and testing.)
        """
        if delta == 0.0:
            return hidden_state
        dim = hidden_state.shape[-1]
        drift = _DistributionalDrift(dim, delta, k=k, seed=seed)
        return drift.apply(hidden_state)

    @staticmethod
    def apply_dist_error(model: nn.Module, delta: float, k: int = 8,
                         seed: int = 42, mode: str = "onset",
                         ramp_steps: int = None, span_trials: bool = False) -> nn.Module:
        """
        State-hook distributional error: drift the brain's internal state
        x -> R x + b during rollout, so an EN trained before the drift is
        evaluated after it. Installs a forward hook on `model` (the mRNN) and
        returns it.

        model must expose .x, .prev_output, .activation_func, .num_neurons, .reset
        (the MichaelsRNN does). Passing an emulator raises ValueError, since only
        the brain has the state manifold that drifts.

        mode:
          "slow"       -> the drift ramps gradually to full magnitude delta,
                          matching the definition's slow, MINDFUL-like timescale.
                          span_trials selects the timescale:
                            span_trials=False (default): within one trial. A small
                              fixed increment per step ramps the state to full delta
                              over `ramp_steps` steps, and the ramp restarts each
                              trial. Set ramp_steps to the trial length.
                            span_trials=True: across trials (multi-session). The
                              session-level drift is applied once per trial (onset)
                              at a magnitude that ramps 0 -> delta over `ramp_steps`
                              total steps spanning trials, so each trial starts more
                              drifted than the last. This models MINDFUL drift that
                              accumulates across sessions rather than resetting.
          "onset"      -> full drift applied once per trial (instantaneous step).
          "persistent" -> full drift applied each step (compounds; not slow).

        Reusing on the same model is safe: any prior drift is removed first.
        """
        for attr in ("x", "prev_output", "activation_func", "num_neurons", "reset"):
            if not hasattr(model, attr):
                raise ValueError(
                    "apply_dist_error expects the brain model (MichaelsRNN) with a "
                    f"'{attr}' attribute; distributional drift acts on the brain "
                    "state, not the emulator. Pass the mRNN, not the EN.")
        if mode not in ("slow", "onset", "persistent"):
            raise ValueError("mode must be 'slow', 'onset', or 'persistent'")
        if mode == "slow" and (ramp_steps is None or ramp_steps < 1):
            raise ValueError("mode='slow' requires ramp_steps >= 1.")

        ErrorInjector.remove_dist_error(model)
        if delta == 0.0:
            return model

        drift = _DistributionalDrift(model.num_neurons, delta, k=k, seed=seed)

        # Per-trial / global state re-armed on reset(). _drift_t is the within-trial
        # step counter (zeroed each trial); _drift_T is the global step counter
        # (persists across trials, for multi-session drift).
        model._drift_pending = (mode == "onset") or (mode == "slow" and span_trials)
        model._drift_t = 0
        model._drift_T = 0
        original_reset = model.reset

        def reset_with_rearm(*args, _orig=original_reset, _m=model, **kwargs):
            out = _orig(*args, **kwargs)
            _m._drift_pending = True     # Re-arm onset / per-session drift
            if not span_trials:
                _m._drift_t = 0          # Restart the within-trial ramp
            return out

        model._drift_original_reset = original_reset
        model.reset = reset_with_rearm

        R_step = b_step = None
        if mode == "slow" and not span_trials:
            R_step, b_step = drift.increment(ramp_steps)

        def _hook(module, _inputs, _output):
            if module.x is None:
                return
            if mode == "onset":
                if not getattr(module, "_drift_pending", False):
                    return
                module._drift_pending = False
                module.x = drift.apply(module.x)
            elif mode == "persistent":
                module.x = drift.apply(module.x)
            elif span_trials:
                # Multi-session: apply session-level drift once per trial (onset),
                # at a magnitude that ramps with global elapsed steps. The global
                # clock advances every step so later trials start more drifted.
                if getattr(module, "_drift_pending", False):
                    module._drift_pending = False
                    frac = min(1.0, module._drift_T / ramp_steps)
                    if frac > 0.0:
                        R, b = drift.transform_at(frac)
                        R = R.to(device=module.x.device, dtype=module.x.dtype)
                        b = b.to(device=module.x.device, dtype=module.x.dtype)
                        module.x = module.x @ R.T + (b if module.x.dim() == 1 else b.unsqueeze(0))
                module._drift_T += 1
            else:
                # Within-trial slow ramp: nudge by one increment per step.
                if module._drift_t >= ramp_steps:
                    return
                module._drift_t += 1
                Rs = R_step.to(device=module.x.device, dtype=module.x.dtype)
                bs = b_step.to(device=module.x.device, dtype=module.x.dtype)
                module.x = module.x @ Rs.T + (bs if module.x.dim() == 1 else bs.unsqueeze(0))
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
        if hasattr(model, "_drift_t"):
            model._drift_t = 0
        if hasattr(model, "_drift_T"):
            model._drift_T = 0
        return model
