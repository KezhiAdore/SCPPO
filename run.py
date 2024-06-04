import sys
sys.path.append('.')
import os
import json
import random
from absl import flags, app, logging
from algo.params import get_hyperparams
from algo.common import set_seed, make_log_name, make_env, make_net, eval, progress
from algo import SCPPO, PPO
from utils.utils import DynamicNormalization
from utils.env_name import mujoco_envs, dmc_envs

FLAGS = flags.FLAGS

flags.DEFINE_string('env', 'dm_control/ball_in_cup-catch-v0', 'Environment Name')
flags.DEFINE_string('device', 'cpu', 'Device Name')
flags.DEFINE_string('agent', 'scppo', 'Agent Name')
flags.DEFINE_string('experiment_id', "base", 'Experiment ID')

flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_integer("history_len", 8, "Off Policy History Length")
flags.DEFINE_float("lam", 0.86, "Lambda")
flags.DEFINE_bool("a_trace", True, "Use a-trace or not")
flags.DEFINE_bool("truncate_traj", False, "Whether to reset env after update agents")

def main(argv):
    log_name = make_log_name(FLAGS.env, FLAGS.seed, FLAGS.agent, FLAGS.experiment_id)
    FLAGS.log_dir = f"./logs/{log_name}/"
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    logging.info("Initialing... log path: {}".format(FLAGS.log_dir))
    logging.get_absl_handler().use_absl_log_file()
    logging.set_verbosity(logging.INFO)
    params = FLAGS.flag_values_dict()
    logging.info("-"*50)
    logging.info("Training Parameters: ")
    logging.info(json.dumps(params, indent=4))
    
    set_seed(FLAGS.seed)
    env = make_env(FLAGS.env)
    # construct network
    obs_space = env.observation_space
    act_space = env.action_space
    
    HP = get_hyperparams(FLAGS.env)
    
    pi_net, v_net = make_net(FLAGS.env, obs_space, HP.hidden_size, act_space, FLAGS.device)
    # choose agent
    if FLAGS.agent == 'ppo':
        agent = PPO(pi_net, v_net, 
                    log_name=log_name, 
                    batch_size=HP.batch_size,
                    batch_num = HP.batch_num,
                    clip_ratio=HP.clip_ratio,
                    )
    elif FLAGS.agent == 'scppo':
        agent = SCPPO(pi_net, v_net, 
                            log_name=log_name, 
                            batch_size=HP.batch_size, 
                            batch_num = HP.batch_num,
                            history_len=FLAGS.history_len, 
                            clip_ratio=HP.clip_ratio,
                            adam_eps=HP.adam_eps,
                            a_trace=FLAGS.a_trace,
                            lam=FLAGS.lam
                            )
    else:
        raise ValueError("Unknown Agent Name")
    
    # record agent hyperparameters
    logging.info('-' * 50)
    logging.info('Agent Hyperparameters:')
    hyperparams = {}
    for key, value in agent.__dict__.items():
        if isinstance(value, (int, float, str)):
            hyperparams[key] = value
    logging.info(json.dumps(hyperparams, indent=4))
    logging.info("-"*50)
    logging.info('Agent: {}'.format(FLAGS.agent))
    logging.info('Environment: {}'.format(FLAGS.env))
    logging.info('Total Steps: {}'.format(HP.total_steps))
    logging.info('Update & Eval Interval: {}'.format(HP.update_interval))
    logging.info("Log Path: {}".format(log_name))
    logging.info("-"*50)
    
    # train
    logging.info('Start training...')
    if (FLAGS.env in mujoco_envs) or (FLAGS.env in dmc_envs):
        dynamic_state = DynamicNormalization(env.observation_space.shape)
    else:
        dynamic_state = None
        
    total_steps = 0
    task_id = progress.add_task("[green]Training", total=HP.total_steps)
    with progress:
        while not progress.finished:
            state, _ = env.reset(seed=random.randint(0, int(1e6)))
            if dynamic_state:
                state = dynamic_state(state)
            done = False
            while not done:
                progress.update(task_id, advance=1)
                total_steps = progress.tasks[task_id].completed
                if total_steps % HP.save_interval == 0:
                    # save agent model
                    agent.save(FLAGS.log_dir, total_steps)
                
                action = agent.choose_action(state)
                # Rescale and perform action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                if dynamic_state:
                    next_state = dynamic_state(next_state)
                
                if total_steps % HP.eval_interval == 0:
                    # eval agent
                    tot_reward = eval(FLAGS.env, agent, dynamic_state)
                    agent.writer.add_scalar('reward', tot_reward, total_steps)
                
                if total_steps % HP.update_interval == 0:
                    # store final transition
                    if FLAGS.truncate_traj:
                        truncated = True
                    agent.store(state, action, reward, next_state, terminated, truncated, info)
                    # update agent
                    loss = agent.update(HP.update_iters)
                    for description, value in loss.items():
                        agent.writer.add_scalar(description, value, total_steps)
                    if FLAGS.truncate_traj:
                        break
                else:
                    agent.store(state, action, reward, next_state, terminated, truncated, info)
                state = next_state

if __name__ == '__main__':
    app.run(main)