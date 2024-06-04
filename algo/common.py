import sys
sys.path.append('.')
import gymnasium as gym
import numpy as np
import torch
import random
from datetime import datetime
from typing import Any
from utils import simple_nets
from utils.env_name import dmc_envs, mujoco_envs
from algo.policy import BasePolicy

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    )

progress = Progress(
    SpinnerColumn(),
    "{task.description}",
    BarColumn(),
    MofNCompleteColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)

class DmcObsWrapper(gym.ObservationWrapper):
    def observation(self, observation: Any) -> Any:
        return np.hstack(list(observation.values()))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def make_log_name(env_name, seed, agent_name, experiment_id):
    now = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
    # env_name = env_name.replace("/", "_")
    return "{}/{}/{}/{}_{}".format(env_name, seed, agent_name, experiment_id, now)

def make_env(env_name:str):
    env = gym.make(env_name)
    if env_name in dmc_envs:
        env = DmcObsWrapper(env)
        obs, _ = env.reset()
        env.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64
        )
    if (env_name in mujoco_envs) or (env_name in dmc_envs):
        env = gym.wrappers.ClipAction(env)
    return env

def make_net(env_name, obs_space, hidden_sizes, act_space, device):
    if env_name in dmc_envs or env_name in mujoco_envs:
        obs_shape = obs_space.shape[0]
        act_dim = act_space.shape[0]
        pi_net = simple_nets.MLPPiNet(obs_shape, hidden_sizes, act_dim).to(device)
        v_net = simple_nets.MLP(obs_shape, hidden_sizes, 1, activate="tanh").to(device)
    else:
        raise ValueError("env_name not found")
    return pi_net, v_net

def eval(env:str, agent:BasePolicy, dynamic_state=None,):
    env = make_env(env)
    tot_reward = 0
    state, _ = env.reset(seed=random.randint(0, int(1e6)))
    done = False
    while not done:
        if dynamic_state:
            state = dynamic_state(state)
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        tot_reward += reward
        state = next_state
        if done:
            break
    return tot_reward
