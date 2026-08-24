import torch
from collections import deque
import math

DEFAULT_ALPHA = 0.1     #target miscoverage: coverage of 90% of predictions with lowest scores
class coverage:
    def __init__(self,max_scores=5000):
        self.scores=deque(maxlen=max_scores)
        self.target = DEFAULT_ALPHA        #static
        self.alpha  = DEFAULT_ALPHA        #the alpha that gets updated



    def coverage_score(self,mu_bar: torch.Tensor, y: torch.Tensor, u: torch.Tensor = None, eps: float = 1e-6):
        s = (mu_bar - y).abs().mean(dim=-1)
        if u is None:
            return s
        return s / (u.clamp_min(1e-6).sqrt() + eps) #Coverage score is (y-mu_bar)/sqrt(u)

    def q(self):
        if not self.scores:
            return math.inf        #nothing measured yet, so nothing counts as a miss
        return torch.quantile(torch.tensor(self.scores), 1 - self.alpha).item() #1-alpha quantile score


    def update(self,current_score,step_size = 0.05):
        if current_score>self.q(): #we ruled out the signal we needed
            self.alpha -= step_size * (1 - self.target)
        else:
            self.alpha += step_size * self.target #tighten the interval
        self.alpha = min(max(self.alpha, 0.0), 1.0) #clamp to between 0 and 1
        self.scores.append(float(current_score)) #append current score

    def calculate_threshold(self, tolerance):
        q = self.q()
        if not math.isfinite(q) or q <= 0.0:
            return math.inf #not calibrated: never halt
        return (tolerance / q) ** 2

def interval_width(cov):
    """Task 4.2.2 -- conformal interval width W_t = 2 * q_hat_{1-alpha_t}.

    The half-width q_hat = cov.q() is the coverage radius, recomputed each step from the
    ACI-updated alpha (subtask 1's `coverage`). Returns math.inf before any score has been
    seen (cov.q() is inf then), which the gate below reads as "maximally uncertain".
    """
    return 2.0 * cov.q()


def gated_amplitude_cap(theta, W_t, W_threshold, safe_norm):
    """Task 4.2.3 -- gated, shrink-only, direction-preserving amplitude cap.

    If the conformal interval W_t exceeds W_threshold, scale theta toward the origin so
    ||theta||_2 <= safe_norm (per-sample scale = min(1, safe_norm / ||theta||)); otherwise
    theta passes through unchanged. The cap ONLY shrinks and never rotates, so the
    co-processor's chosen stimulation direction is preserved while its magnitude is reined
    in exactly when the emulator's prediction interval is wide. Differentiable, so the CPN
    still trains through it (learning direction under the cap).
    """
    if not (W_t > W_threshold):
        return theta
    norm = theta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    scale = (safe_norm / norm).clamp(max=1.0)   # shrink-only: never > 1
    return theta * scale
