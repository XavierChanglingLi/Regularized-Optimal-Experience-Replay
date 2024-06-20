"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import os

import temperature
import critic_net
import policies
from actor import update as update_actor
from actor import update_actor_LABER as update_actor_laber
from critic import target_update, update_v
from critic import update as update_critic
from critic import update_LABER as update_critic_laber
from common import Batch, InfoDict, Model, PRNGKey


# update priority
def update_priority(key: PRNGKey, critic: Model, actor: Model, value: Model, temp: Model,
           batch: Batch, loss_temp: float, double: bool, discount: float, backup_entropy: bool, args={}):

    w = batch.priority.copy()
    beta = args.per_beta
    q1, q2 = critic(batch.observations, batch.actions)
    if double:
        q = jnp.minimum(q1, q2)
    else:
        q = q1
    

    # KL divergence
    if args.per_type == "OER":

        next_v = value(batch.next_observations)
        target_v = batch.rewards + discount * batch.masks * next_v
        current_v = value(batch.observations)
        td_error = target_v - current_v
        a = td_error/loss_temp
        if args.update_scheme == "avg":
            exp_a = jnp.minimum(jnp.exp(a), args.max_clip)
            exp_a = jnp.maximum(exp_a, 1)
            if args.std_normalize:
                exp_a = exp_a / jnp.mean(w * exp_a)
            priority = (beta * exp_a + (1-beta)) * w
        
        elif args.update_scheme == "exp":
            exp_a = jnp.exp(a)
            exp_a = jnp.minimum(exp_a, args.max_clip)
            exp_a = jnp.maximum(exp_a, 1)
            exp_a = jnp.power(exp_a, args.per_alpha)
            if args.std_normalize:
                exp_a = exp_a/jnp.mean(exp_a)
            priority = exp_a

    # Baseline prioritized experience replay
    elif args.per_type == "PER":
        dist = actor(batch.next_observations)
        next_actions = dist.sample(seed=key)
        next_log_probs = dist.log_prob(next_actions)
        next_q1, next_q2 = critic(batch.next_observations, next_actions)
        if double:
            next_q = jnp.minimum(next_q1, next_q2)
        else:
            next_q = next_q1
        target_q = batch.rewards + discount * batch.masks * next_q
        if backup_entropy:
            target_q -= discount * batch.masks * temp() * next_log_probs
        td_error = target_q - q
        a = jnp.absolute(td_error)
        exp_a = jnp.power(a, args.per_alpha)
        priority = exp_a

    else:
        raise ValueError

    # lower bound the weight 
    priority = jnp.maximum(priority, args.min_clip)

    batch.priority = priority

    info = {'priority': priority, 'per_weight': exp_a, 'orig_weight': w, 'per_weight_min': exp_a.min(), 'td_error_mean': td_error.mean()}
    return info

@functools.partial(jax.jit,
                   static_argnames=['backup_entropy', 'update_target', 'policy_update', 'double', 'args'])
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, value: Model, target_critic: Model,
    temp: Model, batch: Batch, discount: float, tau: float, loss_temp: float,
    target_entropy: float, backup_entropy: bool, update_target: bool, policy_update: bool, double: bool, args,
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:


    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            double,
                                            backup_entropy=backup_entropy,
                                            args=args)

    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    value_info = {}
    if args.per_type == "PER":
        new_value = new_critic
    else:
        new_value, value_info = update_v(target_critic, value, batch, loss_temp, discount, double, key, args)
        value = new_value

    per_info = {}
    if args.per:
        per_info = update_priority(key, target_critic, actor, new_value, temp, batch, loss_temp, double, discount, backup_entropy, args=args)

    if policy_update:
        new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch, double)
        new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy)
    else:
        new_actor, actor_info = actor, {}
        new_temp, alpha_info = temp, {}

    return rng, new_actor, new_critic, new_value, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **value_info,
        **alpha_info,
        **per_info
    }

@functools.partial(jax.jit,
                   static_argnames=['backup_entropy', 'update_target', 'policy_update', 'double', 'args'])
def _update_jit_laber(
    rng: PRNGKey, actor: Model, critic: Model, value: Model, target_critic: Model,
    temp: Model, batch: Batch, discount: float, tau: float, loss_temp: float,
    target_entropy: float, backup_entropy: bool, update_target: bool, policy_update: bool, double: bool, args,
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:


    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic_laber(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            double,
                                            backup_entropy=backup_entropy,
                                            args=args)

    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    per_info = {}
    value_info = {}
    new_value = new_critic

    if policy_update:
        new_actor, actor_info = update_actor_laber(key, actor, new_critic, temp, batch, double, args)
        new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy)
    else:
        new_actor, actor_info = actor, {}
        new_temp, alpha_info = temp, {}

    return rng, new_actor, new_critic, new_value, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **value_info,
        **alpha_info,
        **per_info
    }

class SACLearner(object):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 backup_entropy: bool = True,
                 init_temperature: float = 1.0,
                 init_mean: Optional[np.ndarray] = None,
                 policy_final_fc_init_scale: float = 1.0,
                 loss_temp: float = 1.0,
                 double_q: bool = True,
                 args = None):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount
        self.loss_temp = loss_temp
        self.double_q = double_q
        self.args = args

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, temp_key = jax.random.split(rng, 5)
        actor_def = policies.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            init_mean=init_mean,
            final_fc_init_scale=policy_final_fc_init_scale)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        critic_def = critic_net.DoubleCritic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])
        
        if args.per_type == "PER":
            self.value = critic
        else:
            value_def = critic_net.ValueCritic(hidden_dims)
            value = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=optax.adam(learning_rate=value_lr))
            self.value = value

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=temp_lr))

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng

        self.step = 1

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        self.rng = rng

        actions = np.asarray(actions)

        return np.clip(actions, -1, 1)

    def update(self, batch: Batch, policy_update: bool, per_type: str) -> InfoDict:
        self.step += 1

        if per_type == 'LABER':
            new_rng, new_actor, new_critic, new_value, new_target_critic, new_temp, info = _update_jit_laber(
            self.rng, self.actor, self.critic, self.value, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.loss_temp, self.target_entropy,
            self.backup_entropy, self.step % self.target_update_period == 0, policy_update, self.double_q, self.args)
        else:
            new_rng, new_actor, new_critic, new_value, new_target_critic, new_temp, info = _update_jit(
            self.rng, self.actor, self.critic, self.value, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.loss_temp, self.target_entropy,
            self.backup_entropy, self.step % self.target_update_period == 0, policy_update, self.double_q, self.args)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info

    def load(self, save_dir: str):
      self.actor = self.actor.load(os.path.join(save_dir, 'actor'))
      self.critic = self.critic.load(os.path.join(save_dir, 'critic'))
      self.value = self.value.load(os.path.join(save_dir, 'value'))
      self.target_critic = self.target_critic.load(os.path.join(save_dir, 'critic'))

    def save(self, save_dir: str):
      self.actor.save(os.path.join(save_dir, 'actor'))
      self.critic.save(os.path.join(save_dir, 'critic'))
      self.value.save(os.path.join(save_dir, 'value'))
