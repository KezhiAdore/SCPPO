import gymnasium as gym
import sys
import torch
sys.path.append(".")
from utils.simple_nets import MLP, MLPPiNet
from algo import PPO

env = gym.make("CartPole-v1")
obs_shape = env.observation_space.shape[0]
num_actions = env.action_space.n
pi_net = MLPPiNet(obs_shape, [64], num_actions)
v_net = MLP(obs_shape, [64], 1)
pi_optimizer = torch.optim.Adam(pi_net.parameters(), lr=3e-4)
v_optimizer = torch.optim.Adam(v_net.parameters(), lr=3e-4)

def test_ppo():
    agent = PPO(pi_net, v_net, log_name="ppo_test")
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.store(state, action, reward, next_state, terminated, truncated, info)
        state = next_state
        if done:
            break
    agent.update(1)
    
test_ppo()