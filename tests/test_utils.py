import sys
sys.path.append('.')
import torch
import numpy as np
from torch import nn
from utils.utils import *
from utils.simple_nets import MLPPiNet

# test
def test_discount_cum():
    rew = torch.ones((10, 1))
    done = torch.zeros((10, 1))
    done[5] = 1
    gamma = 0.9
    rew = discount_cum(rew, done, gamma)
    assert rew[-1] == 1.0 and rew[5] == 1.0
    assert rew[-2] == 1.9 and rew[3] == 2.71

def test_weight_discount_cum():
    adv = torch.ones((10, 1))
    done = torch.zeros((10, 1))
    done[5] = 1
    weight = torch.ones((10, 1))
    gamma = 0.9
    adv_ = weight_discount_cum(adv, done, weight, gamma)
    assert (adv_ == discount_cum(adv, done, gamma)).all()
    weight = weight * 0.5
    adv_ = weight_discount_cum(adv, done, weight, gamma)
    assert adv_[-1] == 1.0 and adv_[5] == 1.0
    assert adv_[-2] == 1.45 and adv_[3] == 1.6525

def test_v_trace():
    log_rhos = torch.randn(10, 1)
    gamma = 0.9
    rewards = torch.randn(10, 1)
    values = torch.randn(10, 1)
    bootstrap_value = values[-1]
    vs = v_trace(log_rhos, gamma, rewards, values, bootstrap_value)
    assert vs.shape == (10, 1)

def test_dynamic_normalization():
    state = np.random.randn(10, 4)
    dynamic_state = DynamicNormalization(4)
    for i in range(10):
        s = dynamic_state(state[i])
        assert s.shape == (4,)