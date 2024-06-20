from typing import Tuple

import jax
import jax.numpy as jnp
from functools import partial

from common import Batch, InfoDict, Model, Params, PRNGKey


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


def gumbel_rescale_loss(diff, alpha, args):
    # rescale by the norm
    z = diff/alpha
    if args.gumbel_max_clip is not None:
        z = jnp.minimum(z, args.gumbel_max_clip)

    loss = jnp.exp(z) - z - 1

    norm = jnp.mean(jnp.maximum(1, jnp.exp(z)))
    norm = jax.lax.stop_gradient(norm)  # Detach the gradients
    loss = loss / norm
    return loss, norm # jnp.minimum(loss.mean(), 200)

def gumbel_rescale_loss_per(diff, alpha, args):
    x = diff/alpha
    z = jnp.minimum(x, args.gumbel_max_clip)

    # e^x - x - 1                  if x <= d
    # e^d - d - 1 + (e^d - 1) * (x - d)  if x > d
    loss = jnp.exp(z) - z - 1
    # print(B, loss.shape, z.shape)
    linear = (x - z) * (jnp.exp(z) - 1) / alpha

    norm = jnp.mean(jnp.maximum(1, jnp.exp(z)))
    norm = jax.lax.stop_gradient(norm)  # Detach the gradients

    loss = loss + linear
    # log the norm value and check if this is the problem
    loss = loss / norm
    return loss, norm


def update_v(critic: Model, value: Model, batch: Batch,
             loss_temp: float, discount: float, double: bool, key: PRNGKey, args) -> Tuple[Model, InfoDict]:

    obs = batch.observations
    acts = batch.actions

    if args.noise:
        std = args.noise_std
        noise = jax.random.normal(key, shape=(acts.shape[0], acts.shape[1]))
        noise = jnp.clip(noise * std, -0.5, 0.5)
        acts = (batch.actions + noise)
        acts = jnp.clip(acts, -1, 1)

    q1, q2 = critic(obs, acts)
    if double:
        q = jnp.minimum(q1, q2)
    else:
        q = q1

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, obs)
        # check the divergence type 
        if args.per_type == "X2":
            value_loss, norm = conservative_loss(q-v, alpha=loss_temp, args=args)
        elif args.per_type == "OER":
            if args.log_loss:
                value_loss, norm = gumbel_rescale_loss_per(q - v, alpha=loss_temp, args=args)
            else:
                value_loss, norm = gumbel_rescale_loss(q - v, alpha=loss_temp, args=args)
        value_loss = value_loss.mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
            'norm_min': norm.min()
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info

def update_LABER(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, batch: Batch, discount: float, double: bool,
           backup_entropy: bool, args) -> Tuple[Model, InfoDict]:
    
    large_batch_size = args.batch_size
    batch_size = args.mini_batch_size

    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    if double:
        next_q = jnp.minimum(next_q1, next_q2)
    else:
        next_q = next_q1
    
    target_q_large = batch.rewards + discount * batch.masks * next_q

    if backup_entropy:
        target_q_large -= discount * batch.masks * temp() * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1_large, q2_large = critic.apply({'params': critic_params}, batch.observations,
                                 batch.actions)

        td_error1 = jnp.absolute(q1_large - target_q_large)
        probs1 = td_error1/td_error1.sum()
        indices1 = jax.random.choice(key, large_batch_size, shape=(batch_size,), p=probs1)
        td_error_for_selected_indices1 = td_error1[indices1]
        observations_selected1 = batch.observations[indices1]
        actions_selected1 = batch.actions[indices1]
        target_q1 = target_q_large[indices1]
        w1 = (1.0/td_error_for_selected_indices1)*td_error1.mean()
        q1, _ = critic.apply({'params': critic_params}, observations_selected1,
                                 actions_selected1)

        td_error2 = jnp.absolute(q2_large-target_q_large)
        probs2 = td_error2/td_error2.sum()
        indices2 = jax.random.choice(key, large_batch_size, shape=(batch_size,), p=probs2)
        td_error_for_selected_indices2 = td_error2[indices2]
        observations_selected2 = batch.observations[indices2]
        actions_selected2 = batch.actions[indices2]
        target_q2 = target_q_large[indices2]
        w2 = (1.0/td_error_for_selected_indices2)*td_error2.mean()
        _, q2 = critic.apply({'params': critic_params}, observations_selected2,
                                 actions_selected2)

        def mse_loss(q, target_q, w):
            loss_dict = {}
            x = q-target_q
            loss = x**2
            loss_dict['critic_loss'] = (w*loss).mean()
            return (w*loss).mean(), loss_dict

        critic_loss = mse_loss

        loss1, dict1 = critic_loss(q1, target_q1, w1)
        loss2, dict2 = critic_loss(q2, target_q2, w2)

        if double:
            critic_loss = (loss1 + loss2).mean()

        else:
            critic_loss = loss1.mean()

        for k, v in dict2.items():
            dict1[k] += v
        loss_dict = dict1

        if args.grad_pen:
            lambda_ = args.lambda_gp
            q1_grad, _ = grad_norm(critic, critic_params, observations_selected1, actions_selected1)
            _, q2_grad = grad_norm(critic, critic_params, observations_selected2, actions_selected2)
            loss_dict['q1_grad'] = q1_grad.mean()
            loss_dict['q2_grad'] = q2_grad.mean()
            if double:
                gp_loss = (q1_grad + q2_grad).mean()
            else:
                gp_loss = q1_grad.mean()
            critic_loss += lambda_*gp_loss

        loss_dict.update({
            'q1': q1.mean(),
            'q2': q2.mean()
            })
        return critic_loss, loss_dict

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info



def update(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, batch: Batch, discount: float, double: bool,
           backup_entropy: bool, args) -> Tuple[Model, InfoDict]:
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    if double:
        next_q = jnp.minimum(next_q1, next_q2)
    else:
        next_q = next_q1

    target_q = batch.rewards + discount * batch.masks * next_q

    if backup_entropy:
        target_q -= discount * batch.masks * temp() * next_log_probs

    w = batch.priority
    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply({'params': critic_params}, batch.observations,
                                 batch.actions)

        def mse_loss(q, q_target):
            loss_dict = {}
            x = q-q_target
            if args.per==True:
                loss = huber_loss(x, delta = 20.0)
                loss_dict['critic_loss'] = (w*loss).mean()
            else:
                loss = x**2
                loss_dict['critic_loss'] = (w*loss).mean()
            return (w*loss).mean(), loss_dict

        critic_loss = mse_loss

        loss1, dict1 = critic_loss(q1, target_q)
        loss2, dict2 = critic_loss(q2, target_q)
        if double:
            critic_loss = (loss1 + loss2).mean()

        else:
            critic_loss = loss1.mean()

        for k, v in dict2.items():
            dict1[k] += v
        loss_dict = dict1

        if args.grad_pen:
            lambda_ = args.lambda_gp
            q1_grad, q2_grad = grad_norm(critic, critic_params, batch.observations, batch.actions)
            loss_dict['q1_grad'] = q1_grad.mean()
            loss_dict['q2_grad'] = q2_grad.mean()
            if double:
                gp_loss = (q1_grad + q2_grad).mean()
            else:
                gp_loss = q1_grad.mean()
            critic_loss += lambda_*gp_loss

        loss_dict.update({
            'q1': q1.mean(),
            'q2': q2.mean()
            })
        return critic_loss, loss_dict

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info

def grad_norm(model, params, obs, action, lambda_=10):

    @partial(jax.vmap, in_axes=(0, 0))
    @partial(jax.jacrev, argnums=1)
    def input_grad_fn(obs, action):
        return model.apply({'params': params}, obs, action)

    def grad_pen_fn(grad):
        # We use gradient penalties inspired from WGAN-LP loss which penalizes grad_norm > 1
        penalty = jnp.maximum(jnp.linalg.norm(grad, axis=-1) - 1, 0)**2
        return penalty

    # print("print out here",input_grad_fn(obs, action))

    grad1, grad2 = input_grad_fn(obs, action)

    return grad_pen_fn(grad1), grad_pen_fn(grad2)

def huber_loss(x, delta: float = 1.):
    """Huber loss, similar to L2 loss close to zero, L1 loss away from zero.
    See "Robust Estimation of a Location Parameter" by Huber.
    (https://projecteuclid.org/download/pdf_1/euclid.aoms/1177703732).
    Args:
    x: a vector of arbitrary shape.
    delta: the bounds for the huber loss transformation, defaults at 1.
    Note `grad(huber_loss(x))` is equivalent to `grad(0.5 * clip_gradient(x)**2)`.
    Returns:
    a vector of same shape of `x`.
    """
    # 0.5 * x^2                  if |x| <= d
    # 0.5 * d^2 + d * (|x| - d)  if |x| > d
    abs_x = jnp.abs(x)
    quadratic = jnp.minimum(abs_x, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_x - quadratic
    return 0.5 * quadratic**2 + delta * linear

def PAL(x, per_alpha, min_clip):
    return jnp.where(
        jnp.absolute(x) < min_clip,
        (min_clip**per_alpha)*0.5*jnp.power(x, 2),
        min_clip*jnp.power(jnp.absolute(x), 1.0+per_alpha)/(1+per_alpha)
    ).mean()
