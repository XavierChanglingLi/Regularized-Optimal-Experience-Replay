from typing import Tuple

import jax
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params, PRNGKey


def update(key: PRNGKey, actor: Model, critic: Model, temp: Model,
           batch: Batch, double: bool) -> Tuple[Model, InfoDict]:
    
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params}, batch.observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        if double:
            q = jnp.minimum(q1, q2)
        else:
            q = q1
        actor_loss = (log_probs * temp() - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean()
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info



def update_actor_LABER(key: PRNGKey, actor: Model, critic: Model, temp: Model,
           batch: Batch, double: bool, args) -> Tuple[Model, InfoDict]:

    large_batch_size = args.batch_size
    batch_size = args.mini_batch_size
    indices_actor = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=large_batch_size)
    observations_selected_for_actor = batch.observations[indices_actor]

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params}, observations_selected_for_actor)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(observations_selected_for_actor, actions)
        if double:
            q = jnp.minimum(q1, q2)
        else:
            q = q1
        actor_loss = (log_probs * temp() - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean()
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
