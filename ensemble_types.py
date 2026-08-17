"""
ensemble_types.py - the shared interface for the EN ensemble.

Defines the interface every other Stage 2/3 module codes against, so nobody has to import a specific ensemble implementation.
"""
from typing import Protocol, Tuple, runtime_checkable

import torch


@runtime_checkable
class EnsembleLike(Protocol):
    """Any M-member emulator ensemble usable by the trust-aware mechanisms.

    Required attributes and class methods:
      M: int                      number of ensemble members
      forward(x_t) -> (mu, var)   each shape (M, batch, out_dim); per-member
                                  mean and predicted variance for one timestep
      summarize(x_t) -> (mu_bar, aleatoric, epistemic)  each (batch, out_dim)
      reset() -> None             reset all members' recurrent state
    """
    M: int

    def forward(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def summarize(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def reset(self) -> None: ...
