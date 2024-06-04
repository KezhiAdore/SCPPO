import copy
import torch
import numpy as np
from torch import nn, optim, Tensor
from torch.distributions import Categorical, Normal
from typing import Any, List

def discount_cum(rew, done, gamma):
    rew = copy.deepcopy(rew)
    for i in reversed(range(len(rew)-1)):
        rew[i] = rew[i] + gamma * rew[i+1] * (1 - done[i])
    return rew

def weight_discount_cum(adv, done, weight, gamma):
    adv = copy.deepcopy(adv)
    for i in reversed(range(len(adv)-1)):
        adv[i] = adv[i] + weight[i + 1] * gamma * adv[i+1] * (1 - done[i])
    return adv

def minimize_with_clipping(parameters, optimizer:optim.Optimizer, loss:Tensor, max_global_gradient_norm:float):
        optimizer.zero_grad()
        loss.backward()
        if max_global_gradient_norm:
            nn.utils.clip_grad_norm_(parameters, max_global_gradient_norm)
        optimizer.step()

def v_trace(log_rhos:Tensor,
            gamma: float, 
            rewards: Tensor,
            values: Tensor, 
            bootstrap_value: Tensor,
            clip_rho_threshold: float=1.0,
            clip_cs_threshold: float=1.0,
            lam: float=0.95
            ):
    """Compute v-trace targets for PPO.
    
    Espeholt, L. et al. IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures. in Proceedings of the 35th International Conference on Machine Learning 1407–1416 (PMLR, 2018).
    
    Args:
        log_rhos: A float32 tensor of shape [T, B] representing the log importance sampling weights.
        gamma: A float32 scalar representing the discounting factor.
        rewards: A float32 tensor of shape [T, B] representing the rewards.
        values: A float32 tensor of shape [T, B] representing the value function estimates wrt. the
            target policy.
        bootstrap_value: A float32 of shape [B] representing the bootstrap value at time T.
        clip_rho_threshold: A float32 scalar representing clipping threshold for importance weights.
        clip_cs_threshold: A float32 scalar representing clipping threshold for cs values.
    Returns: A float32 tensor of shape [T, B].
    """
    # compute v_next
    # bootstrap_value = torch.zeros_like(bootstrap_value)
    value_next = torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)
    # compute delta V
    rho = torch.exp(log_rhos)
    clipped_rho = torch.clamp(rho, max=clip_rho_threshold)
    delta_V = clipped_rho * (rewards + gamma * value_next - values)
    # compute v trace
    cs = torch.clamp(rho, max=clip_cs_threshold) * lam
    # compute acc
    acc = torch.zeros_like(bootstrap_value)
    result = []
    for t in range(values.shape[0] - 1, -1, -1):
        acc = delta_V[t] + gamma * cs[t] * acc
        result.append(acc)
    result.reverse()
    vs_minus_v_xs = torch.stack(result)
    vs = torch.add(vs_minus_v_xs, values)
    # vs_next = torch.cat([vs[1:], torch.zeros_like(bootstrap_value).unsqueeze(0)], dim=0)
    # adv = rewards + gamma * vs_next - values
    return vs

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape) -> None:
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.ones(shape)
    
    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n )

class DynamicNormalization:
    # Dynamically normalize input
    def __init__(self, shape, baseline=0) -> None:
        self.running_mean_std = RunningMeanStd(shape)
        self.baseline = baseline
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.normalize(*args, **kwds)
    
    def normalize(self, x, update=True):
        if update:
            self.running_mean_std.update(x)
        x = (x - self.running_mean_std.mean) / (self.running_mean_std.std + 1e-8) + self.baseline
        return x

class RolloutNormalization:
    def __init__(self, shape, gamma=0.99) -> None:
        self.returns_rm = RunningMeanStd(shape)
        self.returns = np.zeros(shape)
        self.gamma = gamma
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.normalize(*args, **kwds)
    
    def normalize(self, x):
        self.returns = self.returns * self.gamma + x
        self.returns_rm.update(self.returns)
        x = x / (self.returns_rm.std + 1e-8)
        return x
    
    def reset(self):
        self.returns = np.zeros_like(self.returns)
    
        