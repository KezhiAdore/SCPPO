
from collections import namedtuple
from utils.env_name import mujoco_envs, dmc_envs


Hyperparams = namedtuple('Hyperparams', 
                         ['total_steps', 
                          'update_interval', 
                          'eval_interval', 
                          'save_interval', 
                          'lr', 
                          'adam_eps', 
                          'clip_ratio', 
                          'entropy_coef',
                          'critic_coef',
                          'update_iters', 
                          'batch_size', 
                          'batch_num',
                          'hidden_size',
                          ],
                         )

mujoco_hyperparams = Hyperparams(total_steps=1e6,
                                    update_interval=2048,
                                    eval_interval=2048,
                                    save_interval=5e5,
                                    lr=3e-4,
                                    adam_eps=1e-5,
                                    clip_ratio=0.2,
                                    entropy_coef=0.0,
                                    critic_coef=0.5,
                                    update_iters=10,
                                    batch_size=64,
                                    batch_num=32,
                                    hidden_size=[128,128],
                                    )

dmc_hyperparams = Hyperparams(total_steps=1e6,
                                update_interval=2048,
                                eval_interval=2048,
                                save_interval=5e5,
                                lr=3e-4,
                                adam_eps=1e-5,
                                clip_ratio=0.2,
                                entropy_coef=0.0,
                                critic_coef=0.5,
                                update_iters=10,
                                batch_size=64,
                                batch_num=32,
                                hidden_size=[128,128],
                                )

def get_hyperparams(env_name):
    if env_name in mujoco_envs:
        return mujoco_hyperparams
    elif env_name in dmc_envs:
        return dmc_hyperparams
    else:
        raise ValueError("env_name not found")