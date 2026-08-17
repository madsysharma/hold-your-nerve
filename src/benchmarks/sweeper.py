import torch

def inject_matrix_corruption(
    W_en: torch.Tensor, 
    corruption_level: float, 
    noise_type: str = "gaussian"
) -> torch.Tensor:
    if noise_type == "gaussian":
        delta_p = torch.randn_like(W_en) * corruption_level
    elif noise_type == "zero_out":
        mask = torch.rand_like(W_en) > corruption_level
        return W_en * mask
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")
        
    return W_en + delta_p


def run_experiment_sweep(
    model: torch.nn.Module,
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    loss_fn: torch.nn.Module,
    corruption_levels: list
) -> list:
    model.eval()
    results = []

    with torch.no_grad():
        for noise_level in corruption_levels:
            corrupted_W_en = inject_matrix_corruption(model.W_en, noise_level)
            y_pred, variance = model(input_batch, W_en_override=corrupted_W_en)
            
            loss_val = loss_fn(y_pred, target_batch, variance)
            results.append({
                "corruption_level": noise_level,
                "loss": loss_val.item(),
                "mean_variance": torch.mean(variance).item()
            })
            
    return results
