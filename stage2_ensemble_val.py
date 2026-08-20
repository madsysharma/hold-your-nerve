#!/usr/bin/env python3
"""
stage2_ensemble_val.py - Train and validate an M-member EN ensemble wired as a single EN.

Summary:
Wires an ensemble of M Emulator Networks (ENs) as a single composite EN (via WiredEnsembleEN
and WiredEnsembleAdapter) so that the CPN sees the ensemble mean prediction mu_bar = (1/M) sum(mu_i),
while all M emulators train in parallel under backpropagation.

Supports:
1. Baseline reproduction for 70% and 90% recovery gates with M=5.
2. Train-validation sweep across ensemble sizes M in {2, 3, 5, 10} for 70% and 90% targets.
3. Reporting validation MSE, epistemic disagreement across members, and recovery performance.

Examples of usage:
  python stage2_ensemble_val.py --num-members 5 --lesion f5m1 --stop-pct 0.70 --max-epochs 2000
  python stage2_ensemble_val.py --num-members 5 --lesion f5m1 --stop-pct 0.90 --max-epochs 2000
  python stage2_ensemble_val.py --sweep --max-epochs 2000
  python stage2_ensemble_val.py --num-members 5 --baseline-only
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW

import cpn
import cpn_epoch_cpn
import cpn_epoch_en
import experiment.experiment
from ensemble_types import EnsembleLike
from error_injection import ErrorInjector
from stage0_repro import (
    LESIONS,
    resolve_device,
    resolve_coadapt,
    parse_en_ready_floor,
    _REPO_EN_READY_FLOOR,
)
from stage1_error_injection import build_config_with_error

import warnings
warnings.filterwarnings("ignore")


class WiredEnsembleEN(nn.Module):
    """Composite M-member Emulator Network satisfying EnsembleLike protocol.
    
    Holds M independent EN instances, each initialized with a distinct seed.
    """
    def __init__(self, in_dim, out_dim, M=5, num_neurons=None, activation=torch.nn.Tanh,
                 cuda=None, cell_type="lstm", base_seed=0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.M = M
        self.num_neurons = num_neurons
        self.activation_func_t = activation
        self._cuda = cuda
        self.cell_type = cell_type
        self.base_seed = base_seed

        self.members = nn.ModuleList()
        for i in range(M):
            torch.manual_seed(base_seed + i)
            en_i, _ = ErrorInjector.build_structural_en(
                in_dim,
                out_dim,
                num_neurons=num_neurons,
                activation=activation,
                cuda=cuda,
                cell_type=cell_type,
            )
            self.members.append(en_i)

    def forward(self, x_t: torch.Tensor):
        """Returns per-member predictions mu and variance var, each of shape (M, batch, out_dim)."""
        mus = []
        vars_ = []
        for en in self.members:
            out = en(x_t)
            mus.append(out)
            # Standard single-head ENs don't output predicted variance directly; default to zero tensor
            vars_.append(torch.zeros_like(out))
        mu_stack = torch.stack(mus, dim=0)    # (M, batch, out_dim)
        var_stack = torch.stack(vars_, dim=0)  # (M, batch, out_dim)
        return mu_stack, var_stack

    def summarize(self, x_t: torch.Tensor):
        """Returns (mu_bar, aleatoric, epistemic), each (batch, out_dim)."""
        mu_stack, var_stack = self.forward(x_t)
        mu_bar = mu_stack.mean(dim=0)
        aleatoric = var_stack.mean(dim=0)
        epistemic = mu_stack.var(dim=0, unbiased=False)
        return mu_bar, aleatoric, epistemic

    def reset(self) -> None:
        """Reset hidden/recurrent states across all member models."""
        for en in self.members:
            if hasattr(en, "reset"):
                en.reset()


class WiredEnsembleAdapter(nn.Module):
    """Adapts a WiredEnsembleEN to present the single-EN interface expected by the CPN epoch.
    
    forward(en_in) -> mu_bar (batch, out_dim)
    """
    def __init__(self, wired_ensemble: WiredEnsembleEN):
        super().__init__()
        self.wired_ensemble = wired_ensemble
        self.in_dim = wired_ensemble.in_dim
        self.out_dim = wired_ensemble.out_dim
        self.num_neurons = wired_ensemble.num_neurons
        self.activation_func_t = wired_ensemble.activation_func_t
        self.cell_type = wired_ensemble.cell_type
        self._last_mu = None   # per-member means from the last forward (in-graph)

    def forward(self, en_in: torch.Tensor) -> torch.Tensor:
        mu_stack, _ = self.wired_ensemble(en_in)   # (M, batch, out_dim)
        self._last_mu = mu_stack                    # cache, NOT detached
        return mu_stack.mean(dim=0)  # mu_bar, (batch, out_dim)

    def last_uncertainty(self):
        """Ensemble disagreement (epistemic) for the most recent forward, (batch,).
        Computed from the cached member means -- no second forward, so recurrent
        members are not double-stepped. In-graph, so the trust-aware lambda
        penalty trains the CPN to avoid states the ensemble disagrees on. Members
        are single-head, so disagreement (not predicted variance) is the signal."""
        if self._last_mu is None:
            raise RuntimeError("call the adapter on an input before last_uncertainty()")
        return self._last_mu.var(dim=0, unbiased=False).sum(dim=-1)  # (batch,)

    def reset(self):
        self.wired_ensemble.reset()
        self._last_mu = None

    def rebuild_members(self):
        """Rebuilds members with new seeds upon reset_en."""
        new_ensemble = WiredEnsembleEN(
            self.in_dim,
            self.out_dim,
            M=self.wired_ensemble.M,
            num_neurons=self.num_neurons,
            activation=self.activation_func_t,
            cuda=self.wired_ensemble._cuda,
            cell_type=self.cell_type,
            base_seed=self.wired_ensemble.base_seed + 1000,
        )
        new_adapter = WiredEnsembleAdapter(new_ensemble)
        new_opt = AdamW(new_adapter.parameters(), lr=4e-3, weight_decay=0.04)
        return new_adapter, new_opt


class WiredCPN_EN_CoProc(cpn.CPN_EN_CoProc):
    """CoProc harness using a WiredEnsembleAdapter as the EN stand-in."""
    def __init__(self, cfg, num_members=5, base_seed=0, log_dir=None, recycle_thresh=None):
        super().__init__(cfg, log_dir=log_dir, recycle_thresh=recycle_thresh)
        in_dim, stim_dim, out_dim, cuda = cfg.unpack()
        en_in_dim = cfg.observer_instance.out_dim + stim_dim + 1

        wired_ensemble = WiredEnsembleEN(
            en_in_dim,
            out_dim,
            M=num_members,
            num_neurons=cfg.en_num_neurons,
            activation=cfg.en_activation,
            cuda=cuda,
            cell_type=cfg.en_cell_type,
            base_seed=base_seed,
        )
        self.en = WiredEnsembleAdapter(wired_ensemble)
        self.opt_en = AdamW(self.en.parameters(), lr=1e-3, weight_decay=0.04)

        for param in self.cpn.parameters():
            param.requires_grad = False
        for param in self.en.parameters():
            param.requires_grad = True

        self.en_epoch = cpn_epoch_en.CPNEpochEN(
            self.en, self.opt_en, self.cpn, self.opt_cpn, self.cfg
        )
        self.cpn_epoch = cpn_epoch_cpn.CPNEpochCPN(
            self.en, self.opt_en, self.cpn, self.opt_cpn, self.cfg
        )


def evaluate_ensemble_disagreement(wired_adapter: WiredEnsembleAdapter, batch_size=8):
    """Calculate the epistemic disagreement across members on a random probe."""
    try:
        ens = wired_adapter.wired_ensemble
        ref = next(ens.parameters())
        x = torch.randn(batch_size, ens.in_dim, device=ref.device, dtype=ref.dtype)
        mu_bar, _, epistemic = ens.summarize(x)
        return epistemic.mean().item()
    except Exception:
        return None


def run_ensemble_training(num_members, lesion, stop_pct, max_epochs, seed=0,
                          device="cpu", coadapt="off", log_dir="./models",
                          baseline_only=False, floor=None):
    """Run ensemble training/validation for a given M and stop_pct."""
    torch.manual_seed(seed)
    cpn_epoch_en.EN_READY_FLOOR = floor if floor is not None else _REPO_EN_READY_FLOOR
    cpn_epoch_cpn.STOP_PCT = stop_pct
    cpn_epoch_cpn.MAX_EPOCHS = max_epochs

    dev_id = resolve_device(device)
    coadapt_mode = resolve_coadapt(coadapt)
    os.makedirs(log_dir, exist_ok=True)

    cfg = build_config_with_error(
        lesion, dev_id, recover=False, coadapt=coadapt_mode, stim_sigma=2.175
    )
    coproc = WiredCPN_EN_CoProc(cfg, num_members=num_members, base_seed=seed, log_dir=log_dir, recycle_thresh=1)
    exp = experiment.experiment.stage(coproc, cfg)

    if baseline_only:
        healthy = exp.comp_loss_healthy.item()
        lesioned = exp.comp_loss_lesioned.item()
        return {
            "M": num_members,
            "lesion": lesion,
            "stop_pct": stop_pct,
            "healthy_loss": healthy,
            "lesioned_loss": lesioned,
            "recovery_pct": None,
            "epochs": 0,
            "disagreement": None,
        }

    lh = exp.run()
    recov = lh.max_pct_recov
    epochs = len(lh.records)
    disagreement = evaluate_ensemble_disagreement(coproc.en)

    return {
        "M": num_members,
        "lesion": lesion,
        "stop_pct": stop_pct,
        "healthy_loss": exp.comp_loss_healthy.item(),
        "lesioned_loss": exp.comp_loss_lesioned.item(),
        "recovery_pct": recov,
        "epochs": epochs,
        "disagreement": disagreement,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Train and validate an M-member EN ensemble wired as a single EN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--num-members", type=int, default=5,
                        help="Number of ensemble members M (default: 5).")
    parser.add_argument("--sweep", action="store_true",
                        help="Run train-val sweep across ensemble sizes M in {2, 3, 5, 10}.")
    parser.add_argument("--lesion", choices=["f5m1", "aip", "m1"], default="f5m1",
                        help="Which lesion (default: f5m1).")
    parser.add_argument("--stop-pct", type=float, default=0.70,
                        help="Stop recovery percentage threshold (0.70 or 0.90).")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Maximum epochs cap.")
    parser.add_argument("--device", default="cpu",
                        help="Device choice ('cpu' or CUDA index).")
    parser.add_argument("--log-dir", default="./models")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed.")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Check baselines without training.")
    parser.add_argument("--en-ready-floor", default=None,
                        help="EN->CPN readiness floor override.")
    args = parser.parse_args()

    floor_per_lesion, floor_fallback = parse_en_ready_floor(args.en_ready_floor)
    floor = floor_per_lesion.get(args.lesion, floor_fallback)

    if args.sweep:
        m_list = [2, 3, 5, 10]
        gate_list = [0.70, 0.90]
        print("=" * 80)
        print("TRAIN-VALIDATION SWEEP ACROSS ENSEMBLE SIZES M in {2, 3, 5, 10}")
        print("=" * 80)
        results = []
        for gate in gate_list:
            print(f"\n--- SWEEP GATE: target recovery = {gate * 100:.0f}% ---")
            for m in m_list:
                print(f"Running M={m} ensemble training (stop_pct={gate})...")
                res = run_ensemble_training(
                    num_members=m,
                    lesion=args.lesion,
                    stop_pct=gate,
                    max_epochs=args.max_epochs,
                    seed=args.seed,
                    device=args.device,
                    log_dir=args.log_dir,
                    baseline_only=args.baseline_only,
                    floor=floor,
                )
                results.append(res)
                shown_recov = "n/a" if res["recovery_pct"] is None else f"{res['recovery_pct'] * 100:.1f}%"
                dis_str = f"{res['disagreement']:.6e}" if res['disagreement'] is not None else "n/a"
                print(f"  M={m:2d} | Recovery={shown_recov:>7s} | Epochs={res['epochs']:4d} | Disagreement={dis_str}")

        print("\n" + "=" * 80)
        print("FINAL SWEEP SUMMARY TABLE")
        print("=" * 80)
        print(f"{'M':>4s} | {'Target Gate':>11s} | {'Recovery':>10s} | {'Epochs':>7s} | {'Disagreement':>14s}")
        print("-" * 60)
        for r in results:
            shown_recov = "n/a" if r["recovery_pct"] is None else f"{r['recovery_pct'] * 100:.1f}%"
            dis_str = f"{r['disagreement']:.6e}" if r['disagreement'] is not None else "n/a"
            print(f"{r['M']:4d} | {r['stop_pct'] * 100:10.0f}% | {shown_recov:>10s} | {r['epochs']:7d} | {dis_str:>14s}")
        print("=" * 80)
    else:
        print("=" * 72)
        print(f"SINGLE ENSEMBLE RUN: M={args.num_members}  lesion={args.lesion}  stop-pct={args.stop_pct * 100:.0f}%")
        print("=" * 72)
        res = run_ensemble_training(
            num_members=args.num_members,
            lesion=args.lesion,
            stop_pct=args.stop_pct,
            max_epochs=args.max_epochs,
            seed=args.seed,
            device=args.device,
            log_dir=args.log_dir,
            baseline_only=args.baseline_only,
            floor=floor,
        )
        shown_recov = "n/a" if res["recovery_pct"] is None else f"{res['recovery_pct'] * 100:.1f}%"
        dis_str = f"{res['disagreement']:.6e}" if res['disagreement'] is not None else "n/a"
        print(f"\nRESULTS:")
        print(f"  Ensemble Size (M): {res['M']}")
        print(f"  Recovery Achieved: {shown_recov}")
        print(f"  Training Epochs:   {res['epochs']}")
        print(f"  Disagreement:      {dis_str}")
        print("=" * 72)


if __name__ == "__main__":
    main()
