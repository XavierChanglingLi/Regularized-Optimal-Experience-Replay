
from jax.config import config
import os
from typing import Tuple

import os
import random
import datetime
import gym
import numpy as np
import tqdm
import time
import absl
import sys
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
from dataclasses import dataclass

from sac_learner import SACLearner
from dataset_utils import D4RLDataset, ReplayBuffer
from evaluation import evaluate
from env_utils import make_env

import wandb
import warnings
import wrappers

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'antmaze-umaze-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_string('eval_save_dir', './tmp/evaluation/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 100,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('updates_per_step', 1, 'Gradient updates per step.')
flags.DEFINE_integer('max_steps', int(15e5), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('track', True, 'Track experiments with Weights and Biases.')
flags.DEFINE_string('wandb_project_name', "jaxrl", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', None, "the entity (team) of wandb's project")

# common
flags.DEFINE_float('actor_lr', 3e-4, 'actor learning rate')
flags.DEFINE_float('critic_lr', 3e-4, 'critic learning rate')
flags.DEFINE_float('value_lr', 3e-4, 'value learning rate')
flags.DEFINE_float('temp_lr', 3e-4, 'temperature learning rate')
flags.DEFINE_float('discount', 0.99, 'discount value')
flags.DEFINE_float('tau', 0.005, 'value of tau')
flags.DEFINE_float('target_update_period', 1, 'target network update period')
flags.DEFINE_float('init_temperature', 1.0, 'initial temperature')

# additional for replay buffer
flags.DEFINE_float('max_clip', 50., 'Weight maximum clip value')
flags.DEFINE_float('min_clip', 1., 'Weight minimum clip value')
flags.DEFINE_boolean('double', True, 'Use double q-learning')
flags.DEFINE_boolean('per', True, 'Add PER')
flags.DEFINE_string('per_type', 'OER', 'PER type: X2, OER, PER')
flags.DEFINE_string('update_scheme', 'avg', 'priority update scheme: avg, exp')
flags.DEFINE_boolean('std_normalize', True, 'Noramlize the batch weights by batch mean')
flags.DEFINE_float('per_beta', 0.01, 'PER EMA update rate')
flags.DEFINE_float('per_alpha', 0.4, 'PER adjustment factor')
flags.DEFINE_integer('mini_batch_size', 256, 'LABER update batch size')

flags.DEFINE_integer('capacity', int(2e6), 'Replay buffer capacity')
flags.DEFINE_float('temp', 1.0, 'Loss temperature for priority calculation based on EQL')
flags.DEFINE_boolean('grad_pen', True, 'Add a gradient penalty to critic network')
flags.DEFINE_float('lambda_gp', 1., 'Gradient penalty coefficient')
flags.DEFINE_float('gumbel_max_clip', 7., 'Loss clip value')
flags.DEFINE_boolean('noise', True, 'Add noise to actions for value network')
flags.DEFINE_float('noise_std', 0.1, 'Noise std for actions')
flags.DEFINE_boolean('log_loss', True, 'Use log gumbel loss for value network')

flags.DEFINE_integer('policy_update_delay', 1, 'policy network update delay')

config_flags.DEFINE_config_file(
    'config',
    'configs/sac_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


@dataclass(frozen=True)
class ConfigArgs:
    max_clip: float
    min_clip: float
    per: bool
    per_beta: float
    per_alpha: float
    per_type: str
    update_scheme: str
    std_normalize: bool
    grad_pen: bool
    lambda_gp: float
    gumbel_max_clip: float
    noise: bool
    noise_std: float
    log_loss: bool
    policy_update_delay: int
    batch_size: int
    mini_batch_size: int

def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    eval_env = gym.make(env_name)
    eval_env = wrappers.EpisodeMonitor(eval_env)
    eval_env = wrappers.SinglePrecision(eval_env)
    eval_env.seed(seed+42)
    eval_env.action_space.seed(seed+42)
    eval_env.observation_space.seed(seed+42)


    if 'antmaze' in FLAGS.env_name:
        dataset.rewards -= 1.0
        # dataset.rewards = (dataset.rewards - 0.5) * 4
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        normalize(dataset)

    return env, eval_env, dataset

def main(_):
    kwargs = dict(FLAGS.config)
    algo = kwargs.pop('algo')
    if FLAGS.track:
        wandb.init(project=FLAGS.env_name+algo+'_per_online', sync_tensorboard=True,
               reinit=True, settings=wandb.Settings(_disable_stats=True))
        wandb.config.update(flags.FLAGS)

    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(FLAGS.save_dir, ts_str)
    hparam_str_dict = dict(seed=FLAGS.seed, env=FLAGS.env_name)
    hparam_str = ','.join([
        '%s=%s' % (k, str(hparam_str_dict[k]))
        for k in sorted(hparam_str_dict.keys())
    ])

    summary_writer = SummaryWriter(os.path.join(save_dir, 'tb',
                                                hparam_str),
                                   write_to_disk=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(FLAGS.eval_save_dir, exist_ok=True)

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env, eval_env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
    dataset_size = len(dataset.observations)
    replay_buffer_size = FLAGS.capacity
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 replay_buffer_size)
    replay_buffer.initialize_with_dataset(dataset, dataset_size)


    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    if FLAGS.track:
        wandb.config.update(kwargs)
        wandb.config.update({'base_dir': save_dir})

    args = ConfigArgs(max_clip=FLAGS.max_clip,
                      min_clip=FLAGS.min_clip,
                      per=FLAGS.per, 
                      per_beta=FLAGS.per_beta,
                      per_alpha=FLAGS.per_alpha,
                      per_type=FLAGS.per_type,
                      update_scheme=FLAGS.update_scheme,
                      std_normalize=FLAGS.std_normalize,
                      grad_pen=FLAGS.grad_pen,
                      lambda_gp=FLAGS.lambda_gp,
                      gumbel_max_clip=FLAGS.gumbel_max_clip,
                      noise=FLAGS.noise,
                      noise_std=FLAGS.noise_std,
                      log_loss=FLAGS.log_loss,
                      policy_update_delay=FLAGS.policy_update_delay,
                      batch_size=FLAGS.batch_size,
                      mini_batch_size=FLAGS.mini_batch_size)

    agent = SACLearner(FLAGS.seed,
                       env.observation_space.sample()[np.newaxis],
                       env.action_space.sample()[np.newaxis], 
                       actor_lr = FLAGS.actor_lr,
                       critic_lr = FLAGS.critic_lr,
                       value_lr = FLAGS.value_lr,
                       temp_lr = FLAGS.temp_lr,
                       discount = FLAGS.discount,
                       tau = FLAGS.tau,
                       target_update_period = FLAGS.target_update_period,
                       init_temperature = FLAGS.init_temperature,
                       loss_temp = FLAGS.temp,
                       double_q = FLAGS.double,
                       args=args, 
                       **kwargs)


    best_eval_returns = -np.inf
    eval_returns = []
    observation, done = env.reset(), False

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)
        # reward shaping 
        if 'antmaze' in FLAGS.env_name:
            reward -= 1.0
        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0
        replay_buffer.insert(observation, action, reward, mask, float(done),
                             next_observation)
        observation = next_observation

        if done:
            observation, done = env.reset(), False

            for k, v in info['episode'].items():
                summary_writer.add_scalar(f'training/{k}', v,
                                          info['total']['timesteps'])

            if 'is_success' in info:
                summary_writer.add_scalar(f'training/success',
                                          info['is_success'],
                                          info['total']['timesteps'])

        if i >= FLAGS.start_training:
            for i_update in range(FLAGS.updates_per_step):
                batch = replay_buffer.sample(FLAGS.batch_size)
                update_info = agent.update(batch, (i_update+1) % FLAGS.policy_update_delay == 0,  FLAGS.per_type)

                # update priority
                if args.per and args.per_type != 'LABER':
                    priority = np.asarray(update_info['priority'])
                    replay_buffer.update_priority(batch.indx, priority)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    if v.ndim == 0:
                        summary_writer.add_scalar(f'training/{k}', v, i)
                    else:
                        summary_writer.add_histogram(f'training/{k}', v, i, max_bins=512)
                summary_writer.flush()
                # log statistics of priority 
                priority_dist = replay_buffer.get_priority()
                # mean
                summary_writer.add_scalar(f'training/weights_average', priority_dist.mean(),i)
                # medium
                summary_writer.add_scalar(f'training/weights_median', np.median(priority_dist),i)
                # max
                summary_writer.add_scalar(f'training/weights_max', priority_dist.max(),i)
                # minimum
                summary_writer.add_scalar(f'training/weights_min', priority_dist.min(),i)
                # variance
                summary_writer.add_scalar(f'training/weights_variance', np.var(priority_dist),i)

                summary_writer.flush()


        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                          info['total']['timesteps'])
            summary_writer.flush()

            if eval_stats['return'] >= best_eval_returns:
                best_eval_returns = eval_stats['return']

            summary_writer.add_scalar(f'evaluation/best_returns', best_eval_returns, i)
            wandb.run.summary["best_returns"] = best_eval_returns

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.eval_save_dir, f'{FLAGS.env_name}_{FLAGS.per}_{FLAGS.per_type}_{FLAGS.per_beta}_{FLAGS.gumbel_max_clip}_{FLAGS.temp}_{FLAGS.max_clip}_{FLAGS.min_clip}_{FLAGS.batch_size}_{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])
            agent.save(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}', f'iter_{i}'))
    
    if FLAGS.track:
        wandb.finish()
    sys.exit(0)
    os._exit(0)
    raise SystemExit


if __name__ == '__main__':
    app.run(main)


