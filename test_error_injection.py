import torch
import torch.nn as nn
from error_injection import ErrorInjector


def test_parametric_error_sweep():
    # Range of sigma_p values requested in PR review
    sigma_p_values = [0.0, 0.05, 0.1, 0.2, 0.4, 0.5, 1.0, 1.75, 3.0, 5.0]

    print(f"Testing ErrorInjector.apply_param_error across {len(sigma_p_values)} sigma_p values...\n")

    for sigma_p in sigma_p_values:
        # Create a fresh dummy model for each test value
        model = nn.Linear(100, 50)
        orig_weight = model.weight.clone().detach()
        orig_std = torch.std(orig_weight)

        # Apply parametric error
        ErrorInjector.apply_param_error(model, sigma_p)

        # Calculate injected noise
        delta_w = model.weight - orig_weight
        noise_std = torch.std(delta_w)
        expected_std = sigma_p * orig_std

        # Verify numerical stability (no NaN or Inf)
        assert not torch.isnan(model.weight).any(), f"NaN detected in weights for sigma_p={sigma_p}"
        assert not torch.isinf(model.weight).any(), f"Inf detected in weights for sigma_p={sigma_p}"

        if sigma_p == 0.0:
            assert torch.equal(model.weight, orig_weight), "Weights changed when sigma_p=0.0!"
            print(f"  sigma_p={sigma_p:<4} -> PASS (Weights untouched)")
        else:
            assert not torch.equal(model.weight, orig_weight), f"Weights did not change for sigma_p={sigma_p}"
            assert torch.isclose(noise_std, expected_std, rtol=0.15), (
                f"Noise std mismatch for sigma_p={sigma_p}: Expected ~{expected_std.item():.5f}, got {noise_std.item():.5f}"
            )
            print(
                f"  sigma_p={sigma_p:<4} -> PASS (Expected noise std: {expected_std.item():.5f}, Actual: {noise_std.item():.5f})"
            )

    print("\nSUCCESS: All edge cases and extreme values handled cleanly without errors!")


if __name__ == "__main__":
    test_parametric_error_sweep()