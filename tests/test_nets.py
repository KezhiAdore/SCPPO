import sys
sys.path.append('.')
from utils import simple_nets
import torch
import gymnasium as gym

def test_mlp():
    # test MLP
    mlp = simple_nets.MLP(4, [8, 16], 2)
    input = torch.randn(1, 4)
    out = mlp(input)
    assert out.shape == (1, 2)

def test_dueling():
    # test DuelingNet
    dueling_net = simple_nets.DuelingNet(4, [8, 16], 2)
    input = torch.randn(1, 4)
    out = dueling_net(input)
    assert out.shape == (1, 2)

def test_mujoco():
    # test MujocoMLP
    env = gym.make("Ant-v4")
    obs_shape = env.observation_space.shape[0]
    act_shape = env.action_space.shape[0]
    print(obs_shape, act_shape)
    mujoco_mlp = simple_nets.MLPPiNet(obs_shape, [512,64], act_shape)
    input = torch.randn(obs_shape)
    out = mujoco_mlp(input)
    assert type(out) == torch.distributions.Normal
    assert out.loc.shape[0] == act_shape
    assert out.scale.shape[0] == act_shape
    env.close()
