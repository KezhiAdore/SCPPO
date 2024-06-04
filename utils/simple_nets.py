import torch
from torch import nn
from torch.distributions import normal, categorical
import torch.nn.functional as F
import numpy as np

# This file contains some simple nets for RL algorithms
# MLP: Multi-Layer Perceptron
# DuelingNet: Dueling Network

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MyCategorical(categorical.Categorical):
    def log_prob(self, value):
        log_prob = super().log_prob(value).unsqueeze(-1)
        return log_prob

class MLP(nn.Module):

    def __init__(self, in_size, hidden_sizes, out_size, activate='relu',softmax=False):
        super(MLP, self).__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        self._layers = []
        if activate == 'relu':
            activate_function = nn.ReLU()
        elif activate == 'tanh':
            activate_function = nn.Tanh()
        elif activate == 'sigmoid':
            activate_function = nn.Sigmoid()
        else:
            raise TypeError('activate function should be `relu`, `tanh` or `sigmoid`')
        # hidden layers
        for size in hidden_sizes:
            self._layers.append(
                nn.Sequential(
                    layer_init(nn.Linear(in_features=in_size, out_features=size)),
                    activate_function
                )
            )
            in_size = size
        # output layer
        self._layers.append(
            layer_init(nn.Linear(in_features=in_size, out_features=out_size))
        )
        if softmax:
            self._layers.append(
                nn.Softmax(dim=-1)
            )
        # transfer list to module list
        self._layers = nn.ModuleList(self._layers)

    def forward(self, x):
        out = x
        for layer in self._layers:
            out = layer(out)
        return out

class MLPPiNet(nn.Module):
    # A simple MLP for MuJoCo
    def __init__(self, obs_shape, hidden_sizes, act_shape):
        super(MLPPiNet, self).__init__()
        self.mlp_net = nn.Sequential(
            MLP(obs_shape, hidden_sizes[:-1], hidden_sizes[-1], activate='tanh'),
            nn.ReLU()
        )
        self.mu_net = nn.Sequential(
            nn.Linear(hidden_sizes[-1], act_shape)
        )
        self.log_std = nn.parameter.Parameter(torch.zeros(act_shape))
        self.sigma_net = nn.Sequential(
            nn.Linear(hidden_sizes[-1], act_shape),
            nn.Softplus(),
        )
    
    def forward(self, x):
        x = self.mlp_net(x)
        mu = self.mu_net(x)
        sigma = self.log_std.exp()
        # sigma = self.sigma_net(x)
        dist = normal.Normal(mu, sigma)
        return dist