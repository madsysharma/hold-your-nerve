import torch

import cpn_model
from cpn_utils import CPNENStats, calc_pred_loss, EpochType
from experiment import utils
from error_injection import ErrorInjector


class CPNEpochEN:
    def __init__(self, en, opt_en, cpn, opt_cpn, cfg):
        self.en = en
        self.opt_en = opt_en
        for p in opt_en.param_groups:
            p["lr"] = 4e-3

        self.cpn = cpn
        self.opt_cpn = opt_cpn

        self.cfg = cfg

        self.preds = None
        self.tidx = 0

        # This tracks the number of epochs we've been training the EN
        self.checkpoint_eidx = 0

        self.cpn_noise = None

        self.recent_train_loss = 0.05
        self.recent_pred_loss = 0.05
        self.recent_pred_val_loss = 0.05

        self.reset()

    def reset(self):
        self.preds = None
        self.logvars = []
        self._rel_u = []
        self.tidx = 0
        self.reset_models()

    def reset_period(self):
        self.checkpoint_eidx = 0

    def reset_models(self):
        self.en.reset()
        self.cpn.reset()

        self.cpn_noise = None

        self.opt_en.zero_grad()
        self.opt_cpn.zero_grad()

    def reset_en(self):
        self.recent_pred_val_loss = 0.05
        self.en, self.opt_en = self.new_en(self.en)

        return self.en, self.opt_en

    def new_en(self, old_en):
        # If the EN is a rebuildable wrapper (ensemble adapter, domain-randomized
        # wrapper), rebuild THROUGH it so the wrapper -- and its last_uncertainty /
        # randomization -- survives reset_en. Without this the wrapper is silently
        # replaced by a single vanilla EN and the trust-aware lambda term no-ops.
        if hasattr(old_en, "rebuild_members"):
            return old_en.rebuild_members()
        en, opt_en = ErrorInjector.build_structural_en(
            old_en.in_dim,
            old_en.out_dim,
            num_neurons=old_en.num_neurons,
            activation=old_en.activation_func_t,
            cuda=self.cfg.cuda,
            cell_type=self.cfg.en_cell_type,
        )
        return en, opt_en

    def ensure_noisey_cpn(self, batch_size):
        if self.cpn_noise is None or self.cpn_noise.batch_size != batch_size:
            self.cpn_noise = cpn_model.CPNNoiseyLSTMCollection(
                self.cpn,
                noise_var=1.0 * self.recent_train_loss,
                white_noise_pct=0.3,
                white_noise_var=6,
                cuda=self.cfg.cuda,
            )
            self.cpn_noise.setup(batch_size)

    def ensure_preds(self, batch_size):
        if self.preds is None:
            self.preds = []
            self.logvars = []   # per-step ensemble log-variance (variance-head estimators)
            self._rel_u = []    # per-step ensemble uncertainty at eval (reliability diagrams)

    # Give en_obs_override to use observations from EN with mismatched sigma 
    def forward(self, brain_data, loss_history, is_validation, stim=None, en_obs_override=None):
        batch_size = brain_data[0].shape[0]
        self.ensure_preds(batch_size)

        if stim is None:
            cpn_in = torch.cat(brain_data, axis=1).detach()

            if is_validation:
                new_stim = self.cpn(cpn_in)
            else:
                self.ensure_noisey_cpn(batch_size)
                new_stim = self.cpn_noise(cpn_in)
        else:
            new_stim = stim

        # EN receives (obs, stims, trial_end). Detachment is just in case; we don't want to backprop here)
        # Give en_obs_override to use observations from EN with mismatched sigma 
        new_obs_en = brain_data[0] if en_obs_override is None else en_obs_override
        en_in = torch.cat((new_obs_en, new_stim, brain_data[-1]), axis=1)
        cur_pred = self.en(en_in)

        self.preds.append(cur_pred.unsqueeze(dim=1))
        if hasattr(self.en, "last_logvar_bar"):
            lv = self.en.last_logvar_bar()
            if lv is not None:
                self.logvars.append(lv.unsqueeze(dim=1))
        if is_validation and getattr(self.cfg, "reliability_log_path", None) \
                and hasattr(self.en, "last_uncertainty"):
            # H4: the ensemble's uncertainty for this input, paired below with the
            # emulator's ACTUAL prediction error to test calibration per error axis.
            self._rel_u.append(self.en.last_uncertainty().detach().unsqueeze(dim=1))

        self.tidx += 1

        return new_stim

    def feedback(self, actuals, targets, trial_end, loss_history, is_validation):
        preds = torch.cat(self.preds, axis=1)
        preds = utils.trunc_to_trial_end(preds, trial_end[:, :-1, :])

        pred_loss = calc_pred_loss(preds, actuals)
        if getattr(self, "logvars", None):
            # Train the variance heads to predict mu's squared error (mu detached, so
            # this trains sigma^2 only and leaves the MSE-trained mean untouched).
            logvars = torch.cat(self.logvars, axis=1)
            logvars = utils.trunc_to_trial_end(logvars, trial_end[:, :-1, :])
            resid2 = (actuals - preds.detach()) ** 2
            var_nll = 0.5 * (logvars + resid2 / logvars.exp().clamp_min(1e-6)).mean()
            pred_loss = pred_loss + var_nll

        if is_validation and getattr(self, "_rel_u", None) \
                and getattr(self.cfg, "reliability_log_path", None):
            import json as _json
            u = torch.cat(self._rel_u, axis=1)                                  # (batch, T)
            u = utils.trunc_to_trial_end(u.unsqueeze(-1), trial_end[:, :-1, :]).squeeze(-1)
            err = (preds - actuals).pow(2).sum(dim=-1).sqrt()                   # (batch, T) L2 emulator error
            n = min(u.shape[1], err.shape[1])
            rec = {"u": u[:, :n].reshape(-1).tolist(),
                   "err": err[:, :n].reshape(-1).tolist()}
            with open(self.cfg.reliability_log_path, "w") as _f:
                _json.dump(rec, _f)   # overwrite: keep the latest (best-trained) validation
            self._rel_u = []

        if is_validation:
            self.recent_pred_val_loss = pred_loss.item()
        else:
            self.recent_pred_loss = pred_loss.item()

            back_params = [p for p in self.en.parameters() if p.requires_grad]
            if back_params:                 # oracle EN is the true brain: nothing to train
                pred_loss.backward(inputs=back_params)
                self.opt_en.step()

    def finish(self, loss_history, is_validation, reused_data=False):

        vl = self.recent_pred_val_loss
        for p in self.opt_en.param_groups:
            if vl < 0.0007:
                p["lr"] = 1e-4
            elif vl < 0.005:
                p["lr"] = 3e-3
            else:
                p["lr"] = 4e-3

        last_rec = loss_history.get_recent_record(-2)
        train_loss_out = float("nan")
        train_val_loss_out = float("nan")
        if last_rec is not None:
            last_user_data = last_rec.user_data
            if last_user_data is not None:
                if last_user_data.train_loss == last_user_data.train_loss:
                    self.recent_train_loss = last_user_data.train_loss
                    train_loss_out = last_user_data.train_loss
                train_val_loss_out = last_user_data.train_val_loss

        if vl != vl or vl == float("inf") or vl > 1.5:
            self.en, self.opt_en = self.new_en(self.en)
            en_is_ready = False
            self.reset_period()
        elif (
            vl < max(0.02 * self.recent_train_loss, 0.0003)
            and self.checkpoint_eidx > 100
        ) or self.checkpoint_eidx >= 3000:
            en_is_ready = True
            self.reset_period()
        else:
            self.checkpoint_eidx += 1
            en_is_ready = False

        # Validate/test every 10 epochs
        next_is_validation = (
            not en_is_ready and not is_validation and (self.checkpoint_eidx % 10) == 0
        )

        user_data = CPNENStats(
            "en_offline" if reused_data else "en",
            EpochType.EN_OFFLINE if reused_data else EpochType.EN,
            train_loss_out,
            train_val_loss_out,
            self.recent_pred_loss,
            self.recent_pred_val_loss,
        )

        self.reset()
        return self.en, self.opt_en, next_is_validation, en_is_ready, user_data
