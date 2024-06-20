from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    successes = None
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
        for k in stats.keys():
            stats[k].append(info['episode'][k])

        if 'is_success' in info:
            if successes is None:
                successes = 0.0
            successes += info['is_success']

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    return stats

def value_estimate(agent, observations, Qpos, Qvel, discount: float, env: gym.Env) -> Dict[str, float]:
    stats = {'True_value': []}
    total_mc_return = 0.0
    for i in range(observations.shape[0]):
        _, done = env.reset(), False
        observation = observations[i]
        env.set_state(Qpos[i], Qvel[i])
        mc_return = 0.0
        total_steps = 0
        while not done or total_steps <= 1000:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, reward, done, info = env.step(action)
            mc_return += reward *(discount**total_steps)
            total_steps += 1
        total_mc_return += mc_return
    value_estimate = total_mc_return/(observations.shape[0])
    stats['True_value'] = value_estimate
    return stats
        

