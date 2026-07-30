import torch
import torch.nn as nn
from error_injection import ErrorInjector

# 1. Create a dummy model
model = nn.Linear(100, 50)

# Save a copy of the original weights and compute their standard deviation
orig_weight = model.weight.clone().detach()
orig_std = torch.std(orig_weight)

# 2. Apply parametric error with sigma_p = 0.1
sigma_p = 0.1
ErrorInjector.apply_param_error(model, sigma_p)

# 3. Calculate actual noise injected
delta_w = model.weight - orig_weight
noise_std = torch.std(delta_w)
expected_std = sigma_p * orig_std

print("Original weight Std Dev:", orig_std.item())
print("Expected noise Std Dev :", expected_std.item())
print("Actual noise Std Dev   :", noise_std.item())

# 4. Check assertions
assert not torch.equal(model.weight, orig_weight), "Weights were not updated!"
assert torch.isclose(noise_std, expected_std, rtol=0.1), "Noise standard deviation scale is off!"

print("\nSUCCESS: Parametric error injection math verified!")