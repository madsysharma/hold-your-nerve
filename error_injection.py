import torch

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

    # Set seed so the specific drift manifold is consistent across runs
    g = torch.Generator(device=hidden_state.device)
    g.manual_seed(seed)

    dim = hidden_state.shape[-1]  # 300 units across AIP (100), F5 (100), M1 (100)

    # 1. Generate an orthogonal rotation matrix (q) via QR decomposition
    random_mat = torch.randn(dim, dim, generator=g, device=hidden_state.device)
    q, _ = torch.linalg.qr(random_mat)

    # 2. Scale rotation angle by delta (interpolating from Identity to Q)
    eye = torch.eye(dim, device=hidden_state.device)
    rotation_matrix = (1.0 - delta) * eye + delta * q

    # 3. Create low-dimensional translation shift b scaled by delta
    translation_vector = delta * torch.randn(dim, generator=g, device=hidden_state.device)

    # 4. Apply transformation x_drifted = (x * R) + b
    if hidden_state.dim() == 1:
        drifted_state = torch.matmul(hidden_state, rotation_matrix) + translation_vector
    else:
        drifted_state = torch.matmul(hidden_state, rotation_matrix) + translation_vector.unsqueeze(0)

    return drifted_state