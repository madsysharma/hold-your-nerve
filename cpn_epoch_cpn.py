import torch

from cpn_utils import CPNENStats, calc_pred_loss, calc_train_loss, EpochType
from experiment import lesion, utils
from error_injection import ErrorInjector
STOP_PCT = 0.99
MAX_EPOCHS = None

class CPNEpochCPN:
    def __init__(self, en, opt_en, cpn, opt_cpn, cfg, model_path=None):
        self.en = en
        self.opt_en = opt_en

        self.cpn = cpn
        self.opt_cpn = opt_cpn
        for p in opt_cpn.param_groups:
            p["lr"] = 1e-3

        if model_path is not None:
            cpn.load_state_dict(torch.load(model_path))

        self.cfg = cfg

        self.preds = None
        self.tidx = 0

        # This tracks how many epochs we've been training the CPN
        self.checkpoint_eidx = 0

        self.recent_task_losses = []

        self.recent_train_loss = 0.05
        self.recent_pred_loss = 0.05
        self.recent_train_val_loss = 0.05

        # Weight for stim regularizer
        # self.reg_stim_weight = 2e-8
        self.reg_stim_weight = None
        self.stims = []
        self.uncerts = []   # per-step ensemble uncertainty (in-graph), for the lambda term
        self.recent_apparent_task_loss = None   # eta_hat's task loss (EN preds vs targets)
        self.recent_stim_cost_samples = None    # per-step per-batch stim costs, for CVaR
        self.recent_stim_raw = None             # (batch, T, stim_dim) raw stims, for the damage-proxy fraction

        if cfg.recover_after_lesion:
            self.set_opt_lr = self.lr_sched_aggressive_refine3
            self.cpn_epoch_max_len = 200
        elif (
            type(cfg.lesion_instance) is lesion.LesionType.outputs.value
            and cfg.lesion_instance.start_idx == 0
            and cfg.lesion_instance.end_idx == 50
        ):
            # M1 is an odd beast...
            self.set_opt_lr = self.lr_sched_aggressive_refine3
            self.cpn_epoch_max_len = 100
        else:
            self.set_opt_lr = self.lr_sched_standard
            self.cpn_epoch_max_len = 200

        self.reset()

    def reset(self):
        self.preds = None
        self.uncerts = []
        self.tidx = 0
        self.stims = []
        self.recent_apparent_task_loss = None   # eta_hat's task loss (EN preds vs targets)
        self.recent_stim_cost_samples = None    # per-step per-batch stim costs, for CVaR
        self.recent_stim_raw = None             # (batch, T, stim_dim) raw stims, for the damage-proxy fraction
        self.reset_models()

    def reset_models(self):
        self.en.reset()
        self.cpn.reset()
        self.opt_en.zero_grad()
        self.opt_cpn.zero_grad()

    def reset_period(self):
        self.checkpoint_eidx = 0
        self.recent_task_losses = []

    def set_en(self, en, opt_en):
        self.en = en
        self.opt_en = opt_en
        self.en.reset()
        self.opt_en.zero_grad()

    def ensure_preds(self, batch_size):
        if self.preds is None:
            self.preds = []

    def forward(self, brain_data, loss_history, is_validation):
        batch_size = brain_data[0].shape[0]
        self.ensure_preds(batch_size)

        cpn_in = torch.cat(brain_data, axis=1)

        new_stim = self.cpn(cpn_in)

        if getattr(self.cfg, "conformal_enabled", False):
            # B6 (task 4.2): gate stim amplitude by the ACI conformal interval. Uses
            # subtask 1's coverage (alpha/q_hat), 4.2.2's interval_width, 4.2.3's cap.
            from conformal import coverage as _Coverage, interval_width, gated_amplitude_cap
            if getattr(self, "_coverage", None) is None:
                self._coverage = _Coverage()
            W_t = interval_width(self._coverage)
            new_stim = gated_amplitude_cap(new_stim, W_t,
                                           self.cfg.conformal_width_threshold,
                                           self.cfg.conformal_safe_norm)

        # en receives (obs, stims, trial_end)
        new_obs_en = brain_data[0]
        en_in = torch.cat((new_obs_en, new_stim, brain_data[-1]), axis=1)
        cur_pred = self.en(en_in)

        # Trust-aware: if the EN is an ensemble adapter, capture this step's
        # uncertainty IN-GRAPH so the lambda penalty can train the CPN to avoid
        # states the ensemble disagrees on. A single EN has no last_uncertainty,
        # so uncerts stays empty and the lambda term is simply skipped.
        if hasattr(self.en, "last_uncertainty"):
            self.uncerts.append(self.en.last_uncertainty())

        self.preds.append(cur_pred.unsqueeze(dim=1))

        self.tidx += 1
        self.stims.append(new_stim)

        return new_stim

    def feedback(self, actuals, targets, trial_end, loss_history, is_validation):

        preds = torch.cat(self.preds, axis=1)
        preds = utils.trunc_to_trial_end(preds, trial_end[:, :-1, :])

        if getattr(self.cfg, "conformal_enabled", False) and getattr(self, "_coverage", None) is not None:
            # ACI update: score the emulator's coverage (mu_bar vs actual) and adapt alpha,
            # so the interval (and thus the cap) tightens/loosens over training.
            score = self._coverage.coverage_score(preds, actuals).mean()
            self._coverage.update(float(score.item()), step_size=self.cfg.conformal_step_size)

        pred_loss = calc_pred_loss(preds, actuals)
        train_loss = calc_train_loss(preds, targets)

        # Dual-recovery + safety instrumentation (additive; no effect on training).
        # Apparent task loss eta_hat: the EN's predictions vs the behavioural
        # targets: what the CPN 'sees' through the emulator. Scored with the
        # same calc_task_loss used for true recovery so Delta is apples-to-apples.
        from experiment.stats import calc_task_loss
        self.recent_apparent_task_loss = calc_task_loss(preds, targets).item()
        # Per-step, per-trial stimulation cost samples (squared L2 per step),
        # flattened over steps and batch, for the stim-cost CVaR.
        if self.stims:
            per_step = torch.stack(
                [(st.detach() ** 2).sum(dim=-1) for st in self.stims], dim=0)  # (T, batch)
            self.recent_stim_cost_samples = per_step.flatten()
            # (batch, T, stim_dim): raw amplitudes for charge_damage_fraction.
            self.recent_stim_raw = torch.stack(
                [st.detach() for st in self.stims], dim=1)
        else:
            self.recent_stim_cost_samples = None
            self.recent_stim_raw = None

        if self.reg_stim_weight is not None:
            # Regularizing applied stimulation
            train_loss += self.reg_stim_weight * sum(
                [torch.linalg.norm(s) for s in self.stims]
            )

        # Trust-aware training: L = MSE + ta_lambda*u + ta_beta*C_safety.
        # Gated by cfg; with the defaults (ta_lambda=ta_beta=0) this block is a
        # no-op and training is identical to vanilla B1.
        ta_lambda = getattr(self.cfg, "ta_lambda", 0.0)
        ta_beta = getattr(self.cfg, "ta_beta", 0.0)
        if ta_beta > 0.0 and self.stims:
            from safety import safety_cost
            theta = torch.stack(self.stims, dim=1)          # (batch, T, stim_dim), in-graph
            train_loss = train_loss + ta_beta * safety_cost(theta).mean()
        if ta_lambda > 0.0 and self.uncerts:
            from safety import soft_penalty, hard_halt
            u = torch.stack(self.uncerts, dim=0)             # (T, batch), in-graph
            if getattr(self.cfg, "ta_caution", "soft") == "hard":
                u_term = hard_halt(u, getattr(self.cfg, "ta_halt_threshold", 1.0),
                                   getattr(self.cfg, "ta_halt_penalty", 10.0)).mean()
            else:
                u_term = soft_penalty(u, 1.0).mean()
            train_loss = train_loss + ta_lambda * u_term

        if is_validation:
            self.recent_train_val_loss = train_loss.item()
        else:
            self.recent_train_loss = train_loss.item()
            if not self.cfg.dont_train:
                train_loss.backward(inputs=list(self.cpn.parameters()))

        self.recent_pred_loss = pred_loss.item()

    def lr_sched_aggressive_refine(self, rtl, eidx):
        """
        Args:
            rtl - recent training loss, which we use to determine the learning rate
        """
        if rtl is None or rtl >= 0.005:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-3
        elif rtl >= 0.004:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-4
        elif rtl >= 0.002:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 2e-6
        else:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-6

    def lr_sched_aggressive_refine2(self, rtl, eidx):
        """
        Args:
            rtl - recent training loss, which we use to determine the learning rate
        """
        if rtl is None or rtl >= 0.004:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 5e-4
        elif rtl >= 0.003:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-5
        elif rtl >= 0.002:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 2e-6
        else:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-6

    def lr_sched_aggressive_refine3(self, rtl, eidx):
        """
        Seems to work for M1 co-adapt.
        Args:
            rtl - recent training loss, which we use to determine the learning rate
        """
        if rtl is None or eidx < 4000:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-3
        elif rtl >= 0.008:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-3
        elif rtl >= 0.006:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 5e-4
        elif rtl >= 0.005:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-5
        elif rtl >= 0.004:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 2e-6
        else:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-6

    def lr_sched_standard(self, rtl, eidx):
        """
        Args:
            rtl - recent training loss, which we use to determine the learning rate
        """
        if rtl is None or rtl >= 0.008:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-3
        elif rtl >= 0.006:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 5e-5
        elif rtl >= 0.002:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 2e-6
        else:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-6

    def finish(self, loss_history, is_validation):

        rtl = self.recent_train_loss
        self.set_opt_lr(rtl, loss_history.eidx)

        # Every 10 epochs let's validate/test
        next_is_validation = not is_validation and (loss_history.eidx % 10) == 0

        last_rec = loss_history.get_recent_record(-2)
        pred_val_loss_out = float("nan")
        if last_rec is not None:
            last_user_data = last_rec.user_data
            if last_user_data is not None:
                pred_val_loss_out = last_user_data.pred_val_loss

        rtl = loss_history.recent_task_loss
        if rtl is None:
            # No CPN task loss has been recorded yet, eg: the EN->CPN handoff landed on a validation epoch, where report_val_last_result passes
            # task_loss=None, so recent_task_loss is still None. 
            # Fall back to the CPN train loss so the readiness gate below and the recent_task_losses comparison stay valid instead of raising on None / int.
            rtl = self.recent_train_loss
        self.recent_task_losses.append(rtl)

        _pretrain_out = getattr(self.cfg, "pretrain_en_out", None)
        if _pretrain_out:
            # EN-only mode: reaching the first CPN epoch means the EN is ready/
            # converged (and CPN epochs leave the frozen EN untouched). Save it as
            # a warm-start checkpoint and stop: no CPN training performed.
            import os as _os
            _d = _os.path.dirname(_os.path.abspath(_pretrain_out))
            _os.makedirs(_d, exist_ok=True)
            torch.save(self.en.state_dict(), _pretrain_out)
            print(f"[pretrain-en] saved converged EN to {_pretrain_out}", flush=True)
            we_are_done = True
            en_is_ready = False
        elif (
            loss_history.max_pct_recov > STOP_PCT
            and not self.cfg.dont_train
            and loss_history.lesioned_loss > loss_history.healthy_loss
        ):
            we_are_done = True
            en_is_ready = False
        
        elif MAX_EPOCHS is not None and loss_history.eidx >= MAX_EPOCHS:
            # Hard bound so a non-converging run terminates and still reports whatever recovery it reached.
            we_are_done = True
            en_is_ready = False

        # For now: just run for awhile
        elif self.cfg.dont_train and loss_history.eidx == 250000:
            we_are_done = True
            en_is_ready = False

        else:
            we_are_done = False

            if self.cfg.dont_train:
                en_is_ready = True
                self.reset_period()
            elif self.recent_pred_loss > max(rtl / 10, 6e-4):
                en_is_ready = False
                self.reset_period()
            elif self.checkpoint_eidx >= self.cpn_epoch_max_len:
                en_is_ready = False
                self.reset_period()
            elif self.checkpoint_eidx > 30:
                num_reg = 0
                for l in self.recent_task_losses[-30:]:
                    if l < rtl:
                        num_reg += 1

                if num_reg > 15:
                    en_is_ready = False
                    self.reset_period()
                else:
                    en_is_ready = True
                    self.opt_cpn.step()
                    self.checkpoint_eidx += 1
            else:
                en_is_ready = True
                self.opt_cpn.step()
                self.checkpoint_eidx += 1

        user_data = CPNENStats(
            "cpn",
            EpochType.CPN,
            self.recent_train_loss,
            self.recent_train_val_loss,
            self.recent_pred_loss,
            pred_val_loss_out,
        )

        self.reset()
        return we_are_done, next_is_validation, en_is_ready, user_data