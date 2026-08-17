import torch
import torch.nn as nn


class TrustAwareLoss(nn.Module):
    """Heteroscedastic Trust-Aware Loss for Emulator Networks (EN)."""
    def __init__(self, trust_threshold: float = 0.05, eps: float = 1e-6):
        super().__init__()
        self.trust_threshold = trust_threshold
        self.eps = eps

    def forward(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor, 
        variance: torch.Tensor
    ) -> torch.Tensor:
        var_clamped = torch.clamp(variance, min=self.eps)
        precision = 1.0 / (2.0 * var_clamped)
        squared_error = (y_true - y_pred) ** 2
        
        loss = precision * squared_error + 0.5 * torch.log(var_clamped)
        return torch.mean(loss)


class TrustAwareCPNLoss(nn.Module):
    """Composite loss for CPN, coupling trust-weighted error with control effort."""
    def __init__(self, lambda_u: float = 0.01, lambda_trust: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.trust_loss = TrustAwareLoss(eps=eps)
        self.lambda_u = lambda_u
        self.lambda_trust = lambda_trust

    def forward(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor, 
        u_control: torch.Tensor, 
        variance: torch.Tensor
    ) -> torch.Tensor:
        l_trust = self.trust_loss(y_pred, y_true, variance)
        l_control = torch.mean(torch.sum(u_control ** 2, dim=-1))
        return self.lambda_trust * l_trust + self.lambda_u * l_control
