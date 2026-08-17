import torch
from typing import Dict


def activation_tanh_derivative(x: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.tanh(x) ** 2


def activation_rectified_tanh_derivative(z: torch.Tensor) -> torch.Tensor:
    t = torch.tanh(z)
    return torch.where(z > 0, 1.0 - t ** 2, torch.zeros_like(z))


def compute_system_jacobian(
    x_star: torch.Tensor,
    z_star: torch.Tensor,
    W_rec: torch.Tensor,
    W_z: torch.Tensor,
    W_x: torch.Tensor,
    W_en: torch.Tensor,
    tau_x: float,
    tau_z: float
) -> torch.Tensor:
    N = x_star.shape[0]
    M = z_star.shape[0]

    d_phi = activation_tanh_derivative(x_star)
    d_psi = activation_rectified_tanh_derivative(z_star)

    J11 = - (1.0 / tau_x) * torch.eye(N, device=x_star.device) + W_rec * d_phi.unsqueeze(0)
    J12 = W_z
    J21 = W_x
    J22 = - (1.0 / tau_z) * torch.eye(M, device=z_star.device) + W_en * d_psi.unsqueeze(0)

    top_row = torch.cat([J11, J12], dim=1)
    bottom_row = torch.cat([J21, J22], dim=1)
    return torch.cat([top_row, bottom_row], dim=0)


def evaluate_nullcline_stability(J_sys: torch.Tensor) -> Dict[str, torch.Tensor]:
    eigenvalues = torch.linalg.eigvals(J_sys)
    real_parts = eigenvalues.real
    max_real_eigenvalue = torch.max(real_parts)
    is_stable = max_real_eigenvalue < 0.0

    return {
        "is_stable": is_stable,
        "max_real_eigenvalue": max_real_eigenvalue,
        "eigenvalues": eigenvalues
    }
