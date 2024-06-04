import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC,TD3,PPO,A2C
from sb3_contrib import TRPO
from absl import flags, app, logging
import random
from algo.common import set_seed

class ObsWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return np.hstack(list(observation.values()))

FLAGS = flags.FLAGS

flags.DEFINE_string('env', "Reacher-v4", 'Environment Name')
flags.DEFINE_string('device', 'auto', 'Device Name')
flags.DEFINE_string("algo", 'sac', 'algorithm name')
flags.DEFINE_integer("seed", 0, 'random seed')

algo_dict = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "trpo": TRPO
}


def main(argv):
    set_seed(FLAGS.seed)
    
    env = gym.make(FLAGS.env)
    if "dm_control" in FLAGS.env:
        env = ObsWrapper(env)
        obs,_ = env.reset(seed=random.randint(0, int(1e6)))
        env.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64
        )
    model = algo_dict[FLAGS.algo](
        "MlpPolicy", 
        env, 
        tensorboard_log=f"logs/{FLAGS.env}/{FLAGS.seed}/{FLAGS.algo}", 
        stats_window_size=1,
        device=FLAGS.device,
        seed=FLAGS.seed
        )
    model.learn(1e6, progress_bar=True)

if __name__ == '__main__':
    app.run(main)
