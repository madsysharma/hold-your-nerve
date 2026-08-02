import torch
import torch.nn as nn

class ErrorInjector: 
    """Unified Interface for Injecting Emulator Errors."""
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

    @staticmethod
    def apply_distributional_error(hidden_state: torch.Tensor, delta: float, seed: int = 42) -> torch.Tensor:
        """
        Injects distributional error by applying a low-dimensional rotation and 
        translation to the mRNN latent state space (x_t -> R * x_t + b).
        
        Args:
            hidden_state: Tensor representing the mRNN hidden state across AIP, F5, and M1.
            delta: Drift magnitude parameter (sweep range: 0.0 to 0.4).
            seed: Random seed for reproducibility of the drift direction.
        """
        if delta == 0.0:
            return hidden_state

        g = torch.Generator(device=hidden_state.device)
        g.manual_seed(seed)

        dim = hidden_state.shape[-1]  

        random_mat = torch.randn(dim, dim, generator=g, device=hidden_state.device)
        q, _ = torch.linalg.qr(random_mat)

        eye = torch.eye(dim, device=hidden_state.device)
        rotation_matrix = (1.0 - delta) * eye + delta * q

        translation_vector = delta * torch.randn(dim, generator=g, device=hidden_state.device)

        if hidden_state.dim() == 1:
            drifted_state = torch.matmul(hidden_state, rotation_matrix) + translation_vector
        else:
            drifted_state = torch.matmul(hidden_state, rotation_matrix) + translation_vector.unsqueeze(0)

        return drifted_state