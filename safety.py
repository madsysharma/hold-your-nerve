"""
safety.py

Defines the safety cost C_safety(theta) and the two caution styles the trust-aware loss can use.

The pieces that are fully determined (stimulation energy; the soft and hard caution functions) are already implemented:
the physical damage proxy and the out-of-distribution fraction are to be filled in by .
"""
from dataclasses import dataclass
from typing import Optional
import math
import torch
from ensemble_types import EnsembleLike


# Shannon damage threshold. Above this, histology shows tissue damage.
SHANNON_K_THRESHOLD = 1.85

#Utah-array microelectrode defaults
DEFAULT_PULSE_WIDTH_S = 200e-6  #200 us/phase
DEFAULT_ELECTRODE_GSA_CM2 = 2.0e-5  #~2000 um^2 exposed tip
DEFAULT_UA_PER_UNIT = 100.0 #microamps per dimensionless CPN unit, this is a default that gets overridden!
#calibrate to find the correct ua per unit

# Cogan's microelectrode ceiling, ~4 nC per phase, expressed in microcoulombs so
# it compares directly against charge_per_phase_uc. Unlike Shannon's k this is a
# flat cap: it does not trade charge off against charge density.
DEFAULT_COGAN_LIMIT_UC = 4e-3

DAMAGE_MODES = ("shannon", "cogan")


@dataclass
class SafetyWeights:
    """Relative weights for clubbing the safety terms into one cost."""
    energy: float = 1.0
    charge: float = 0.0     # Set > 0 to switch on charge_damage_fraction
    ood: float = 0.0        # Set > 0 to switch on ood_fraction


@dataclass(frozen=True)
class StimCalibration:
    #Maps dimensionless CPN stim amplitudes onto physical stimulation units

    ua_per_unit: float = DEFAULT_UA_PER_UNIT
    pulse_width_s: float = DEFAULT_PULSE_WIDTH_S
    electrode_gsa_cm2: float = DEFAULT_ELECTRODE_GSA_CM2
    cogan_limit_uc: float = DEFAULT_COGAN_LIMIT_UC

    def current_ua(self, amp): #amplitude of CPN stim to ua
        return amp.abs() * self.ua_per_unit

    def charge_per_phase_uc(self, amp): #Total charge delivered per phase
        return self.current_ua(amp) * self.pulse_width_s

    def shannon_k(self, amp, eps=1e-12): #k = log10(D) + log10(Q). Undefined at Q=0, so clamp (clamp value is arbitrary)
        q = self.charge_per_phase_uc(amp).clamp_min(eps)
        return 2.0 * torch.log10(q) - math.log10(self.electrode_gsa_cm2)

    @property
    def amp_at_k_threshold(self): #Dimensionless "current" at which Shannon k hits SHANNON_K_THRESHOLD
        return self.amp_at_k(SHANNON_K_THRESHOLD)

    def amp_at_k(self, k): #amplitude where we get a given k, "maximum" current in our phase without crossing k line
        q_star = math.sqrt(self.electrode_gsa_cm2 * (10.0 ** k))  #charge (uC) per phase at k
        return q_star / (self.ua_per_unit * self.pulse_width_s) #divide by time and conversion factor to get back to dimensionless units

    @property
    def amp_at_cogan_limit(self): #dimensionless amplitude at Cogan's ~4 nC/phase ceiling
        return self.cogan_limit_uc / (self.ua_per_unit * self.pulse_width_s)

    @classmethod
    def calibrate_ua_per_unit(
        cls, #StimCalibration class
        reference_amp,  #99th percentile amplitude of unconstrained CPN, dimensionless
        target_k=SHANNON_K_THRESHOLD,
        pulse_width_s=DEFAULT_PULSE_WIDTH_S,
        electrode_gsa_cm2=DEFAULT_ELECTRODE_GSA_CM2,
        **kwargs,
    ):
        q_star = math.sqrt(electrode_gsa_cm2 * (10.0 ** target_k)) #from shannon inequality, find max charge per phase at target_k, uC/phaase
        ua_per_unit = q_star / (abs(reference_amp) * pulse_width_s) #divide target max charge by max charge in CPN units to get conversion factor
        return cls(
            ua_per_unit=ua_per_unit,
            pulse_width_s=pulse_width_s,
            electrode_gsa_cm2=electrode_gsa_cm2,
            **kwargs,
        )

    def render(self):
        return{
            "ua_per_unit": self.ua_per_unit,
            "pulse_width_s": self.pulse_width_s,
            "electrode_gsa_cm2": self.electrode_gsa_cm2,
            "cogan_limit_uc": self.cogan_limit_uc,
            "amp_at_k_threshold": self.amp_at_k_threshold,
            "amp_at_cogan_limit": self.amp_at_cogan_limit,
        }


def _as_steps(x):
    #Normalise (batch, D) -> (batch, 1, D)
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"expected (batch, D) or (batch, T, D), got shape {tuple(x.shape)}")


def stim_energy(theta: torch.Tensor) -> torch.Tensor:
    """Stimulation energy per batch element: sum of squared stimulation.
    theta: (batch, n_channels) for one step, or (batch, T, n_channels).
    Returns (batch,)."""
    dims = tuple(range(1, theta.dim()))
    return (theta ** 2).sum(dim=dims)


def charge_damage_fraction(theta: torch.Tensor, k_threshold: float = SHANNON_K_THRESHOLD,*, calib: Optional[StimCalibration] = None,mode: str = "shannon"):
    #shannon: k >= k_threshold
    #cogan:   Q >= ~4 nC per phase 
    #theta = stimulation channel tensor for each trial at a given timestep
    calib = calib or StimCalibration()
    if mode == "shannon":
        limit = calib.amp_at_k(k_threshold)
    elif mode == "cogan":
        limit = calib.amp_at_cogan_limit
    else:
        raise ValueError(f"unknown damage mode: {mode}")

    over = _as_steps(theta).abs() >= limit           #(batch, T, C)
    return over.any(dim=-1).float().mean(dim=1)      #.any() returns true if any channel is over threshold


def predictive_width(ensemble: EnsembleLike, x_t: torch.Tensor, z: float = 1.96) -> torch.Tensor:
    #how wide is the range of muscle activations plausible for this trial (batch,).
    mu, var = ensemble.forward(x_t)                             #(M, batch, out_dim)
    total_var = var.mean(dim=0) + mu.var(dim=0, unbiased=False) #aleatoric + epistemic
    return (2.0 * z * total_var.clamp_min(0.0).sqrt()).mean(dim=-1)


class SupportModel: #predict if a (neural_activity,stimulation) is OOD or not. In statistics, support is the region where we have data

    DEFAULT_MIN_SAMPLES = 2000 #minimum number of samples to fit the support model, otherwise we don't have enough data to estimate mean and std
    DEFAULT_RESERVOIR_SIZE = 5000 #raw rows kept for the empirical threshold. 5000 x 37 x 8B = 1.5MB
    DEFAULT_ROWS_PER_OBSERVE = 8 #cap per call, observe() runs every EN timestep for thousands of epochs

    def __init__(self, dim, device=None, quantile=0.95, threshold=None): #dim is x_t.shape[-1], number of features in the input
    #dim = recorded mRNN activity, stimulus, and trial end flag. We want to see if this is OOD or not. State action pair = (neural_activity,stimulation)
        self._sum   = torch.zeros(dim, dtype=torch.float64, device=device)
        self._sumsq = torch.zeros(dim, dim, dtype=torch.float64, device=device)
        self.n = 0
        self.quantile = quantile #which percentile of training scores becomes the cutoff
        self.threshold = threshold #None = measure it in fit(). Pass a number to override
        self._mean = torch.zeros(dim, dtype=torch.float64, device=device)
        self._std = torch.ones(dim, dtype=torch.float64, device=device)
        self.min_samples = self.DEFAULT_MIN_SAMPLES
        self.is_fitted = False
        self._reservoir = None #uniform sample of rows, needed because a quantile can't come from sum/sumsq
        self._filled = 0 #slots used
        self._seen = 0 #rows ever offered, drives the replacement probability

    def observe(self, x_t): #record data shape from EN phase for OOD detection, call every timestep
        x = x_t.detach().to(torch.float64) #for one timestep
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        self._sum += x.sum(0)
        self._sumsq += x.T @ x
        self.n += x.shape[0] #number of rows tracked so far
        self._reservoir_add(x)

    def _reservoir_add(self, x): #Algorithm R, samples uniformly from a population with unknown size (reservoir is the collection of samples)
        size = self.DEFAULT_RESERVOIR_SIZE
        take = min(self.DEFAULT_ROWS_PER_OBSERVE, x.shape[0])
        rows = x[torch.randperm(x.shape[0], device=x.device)[:take]]

        if self._reservoir is None:
            self._reservoir = torch.zeros(size, x.shape[1], dtype=torch.float64, device=x.device)
        res = self._reservoir

        if self._filled < size: #fill phase, everything goes in while there is room
            k = min(size - self._filled, rows.shape[0])
            res[self._filled:self._filled + k] = rows[:k]
            self._filled += k
            self._seen += k
            rows = rows[k:]
            if rows.shape[0] == 0:
                return

        m = rows.shape[0] #replacement phase, row i survives with probability size/seen
        ns = self._seen + torch.arange(1, m + 1, device=x.device, dtype=torch.float32)
        j = (torch.rand(m, device=x.device) * ns).long()
        keep = j < size
        if keep.any():
            res[j[keep]] = rows[keep]
        self._seen += m

    def fit(self): #compute mean and std from recorded data
        if self.n < self.min_samples:
            return False
        self._mean = self._sum / self.n
        var = torch.diagonal(self._sumsq) / self.n - self._mean ** 2
        self._std = var.clamp(min=1e-6).sqrt()
        if self.threshold is None and self._filled > 0: #measured, not assumed. Recompute from raw rows so it matches the mean/std just set
            self.threshold = torch.quantile(self.score(self._reservoir[:self._filled]), self.quantile).item() #Z score technically doesnt imply normality, can still use to find percentiles
        self.is_fitted = True
        return True

    def score(self, x_t): #if z score is above threshold, then OOD. Returns z score for each row in x_t
        x = x_t.detach().to(torch.float64).to(self._mean.device)
        z = (x - self._mean) / self._std
        return z.norm(dim=-1).to(torch.float32).to(x_t.device) #euclidean distance along rows

    def ood_mask(self, x_t):
        if not self.is_fitted or self.threshold is None:
            return torch.zeros(x_t.shape[:-1], dtype=torch.bool, device=x_t.device)
        return self.score(x_t) > self.threshold

    def normalised(self, x_t): #continuous version. 1.0 sits exactly on the boundary
        if not self.is_fitted or self.threshold is None:
            return torch.zeros(x_t.shape[:-1], device=x_t.device)
        return self.score(x_t) / max(float(self.threshold), 1e-8)

    def render(self):
        return {
            "support_quantile": self.quantile,
            "support_threshold": self.threshold,
            "support_rows_seen": self.n,
            "support_is_fitted": self.is_fitted,
        }


def ood_fraction(ensemble: Optional[EnsembleLike], x_t: torch.Tensor,
                 width_threshold: float, *,
                 support: Optional[SupportModel] = None, z: float = 1.96):
    #Fraction of steps where the model is out of its trusted region.
    X = _as_steps(x_t) #(batch, T, in_dim)

    if support is not None:
        return support.ood_mask(X).float().mean(dim=1)

    if ensemble is None:
        raise ValueError("ood_fraction needs an ensemble, or support=<fitted SupportModel>.")

    #forward() takes one timestep at a time, and members are recurrent
    #wherever predictive width>width_threshold for a trial at a timestep, add a true flag, otherwise false. Stack flags and see how often we were OOD in a given trial
    flags = []
    for t in range(X.shape[1]):
        flags.append(predictive_width(ensemble, X[:, t], z=z)>width_threshold)
    return torch.stack(flags, dim=1).float().mean(dim=1)


def safety_cost(theta: torch.Tensor, *, ensemble: Optional[EnsembleLike] = None,
                x_t: Optional[torch.Tensor] = None,
                weights: Optional[SafetyWeights] = None,
                calib: Optional[StimCalibration] = None, damage_mode: str = "shannon",
                support: Optional[SupportModel] = None,
                width_threshold: float = 1.0):
    #Combined safety cost C_safety for a given trial

    weights = weights or SafetyWeights()
    cost = weights.energy * stim_energy(theta)
    if weights.charge > 0.0:
        cost = cost + weights.charge * charge_damage_fraction(
            theta, calib=calib, mode=damage_mode)
    if weights.ood > 0.0:
        if x_t is None:
            raise ValueError("weights.ood > 0 needs x_t= (the emulator inputs to score).")
        cost = cost + weights.ood * ood_fraction(
            ensemble, x_t, width_threshold=width_threshold, support=support)
    return cost


# ---- Caution styles (how uncertainty turns into a penalty / action) --------
def soft_penalty(u: torch.Tensor, lam: float) -> torch.Tensor:
    """MOPO-style graded penalty: lam * u.
    Returns same shape as u."""
    return lam * u


def hard_halt(u: torch.Tensor, threshold: float, penalty: float) -> torch.Tensor:
    """MOReL-style hard cutoff: a large penalty wherever u exceeds threshold, zero elsewhere.
    Returns same shape as u. (The paired action - clamping theta
    to zero on HALT - is applied at the call site in the training loop.)"""
    return torch.where(u > threshold,
                       torch.full_like(u, penalty),
                       torch.zeros_like(u))


def halt_mask(u: torch.Tensor, threshold: float, latch: Optional[torch.Tensor] = None):
    #hard_halt: which trials should stop being stimulated
    over = u > threshold
    return over if latch is None else (latch | over)
