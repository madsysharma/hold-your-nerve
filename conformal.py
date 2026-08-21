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

    