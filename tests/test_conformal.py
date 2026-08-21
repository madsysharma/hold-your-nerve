"""
Contract tests for the Gibbs & Candes ACI loop. Pure numbers, no model and no experiment stack.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import torch

from conformal import DEFAULT_ALPHA, coverage


def test_nothing_measured_yet_is_not_a_miss():
    c = coverage()
    assert c.q() == math.inf
    c.update(torch.tensor(0.5))
    assert len(c.scores) == 1
    assert c.alpha > DEFAULT_ALPHA          #inf means covered, so alpha steps up


def test_the_step_is_asymmetric():
    #covered: +gamma*alpha, missed: -gamma*(1-alpha). At alpha=0.1 a miss should move nine times
    #as far as a hit
    hit = coverage()
    hit.scores.extend([1.0] * 100) #q is exactly 1.0
    hit.update(torch.tensor(0.0)) #point is perfectly on target
    assert abs(hit.alpha - 0.105) < 1e-9

    miss = coverage()
    miss.scores.extend([1.0] * 100)
    miss.update(torch.tensor(5.0)) #point is off target
    assert abs(miss.alpha - 0.055) < 1e-9


def test_alpha_stays_in_the_unit_interval():
    low = coverage()
    for i in range(200):
        low.update(torch.tensor(float(i + 1))) #always one step past the window, conformal has never seen this value before so it is above 100th quantile
    assert low.alpha == 0.0

    high = coverage()
    for _ in range(400):
        high.update(torch.tensor(0.0)) #cover every time
    assert high.alpha == 1.0

#alpha too low -> too many points allowed through window -> alpha increases, and vice versa
def test_a_constant_shift_gets_absorbed():
    c = coverage()
    c.scores.extend([1.0] * 100)
    for i in range(200):
        c.update(torch.tensor(1e6)) #alpha goes down from 1 mil to 0
    assert c.q() == 1e6 #the new level is inside the interval now so alpha_t has been climbing back
    #after   1 updates      1 are 1e6 ( 1%)  alpha=0.055  q=1
    #after   2 updates      2 are 1e6 ( 2%)  alpha=0.010  q=989998
    #after   3 updates      3 are 1e6 ( 3%)  alpha=0.000  q=1000000
    #after   4 updates      4 are 1e6 ( 4%)  alpha=0.005  q=1000000
    #after 200 updates    200 are 1e6 (67%)  alpha=0.635  q=1000000

    assert c.alpha > 0.0


def test_coverage_converges_on_the_target():
    #10% miss rate on a stationary stream regardless of scores 
    torch.manual_seed(0)
    c = coverage(max_scores=200)
    missed = 0
    for _ in range(6000):
        s = torch.randn(1).abs()
        if s.item() > c.q():
            missed += 1
        c.update(s)
    assert 0.06 <= missed / 6000 <= 0.14 #reasonable range i think


def test_coverage_recovers_after_a_drift():
    #CPN phase shifts the EN's inputs, errors jump, alpha_t falls and q_t grows until coverage comes back.
    torch.manual_seed(0)
    c = coverage(max_scores=200)
    for _ in range(2000):
        c.update(torch.randn(1).abs())
    q_before = c.q()

    missed = 0
    for _ in range(4000):
        s = 20.0 * torch.randn(1).abs()
        if s.item() > c.q():
            missed += 1
        c.update(s)

    assert c.q() > 10.0 * q_before          #widened to fit the new scale
    assert 0.06 <= missed / 4000 <= 0.18    #and coverage came back to target


def test_score_divides_by_sqrt_u_not_u():
    c = coverage()
    mu_bar = torch.tensor([[1.0, 3.0]])
    y = torch.zeros(1, 2)

    assert torch.allclose(c.coverage_score(mu_bar, y), torch.tensor([2.0])) #no u: plain error

    u = torch.tensor([4.0])
    assert torch.allclose(c.coverage_score(mu_bar, y, u),
                          torch.tensor([1.0]), atol=1e-5) #2/sqrt(4), not 2/4

    #quadrupling u should halve the score
    assert torch.allclose(c.coverage_score(mu_bar, y, torch.tensor([16.0])),
                          c.coverage_score(mu_bar, y, u) / 2.0, atol=1e-5)

