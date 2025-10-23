from functools import partial
from typing import NamedTuple

import jax


class MCMCUpdateInfo(NamedTuple):
    """Thin layer to hold all the info pertaining to the update step."""

    mcmc_states: NamedTuple
    mcmc_infos: NamedTuple


def update_with_mcmc_take_last(
    constrained_mcmc_step_fn,
    num_mcmc_steps,
):
    """An update strategy for NS that uses MCMC to update the particles."""

    def update_function(rng_key, state, loglikelihood_0, step_parameters):
        shared_mcmc_step_fn = partial(
            constrained_mcmc_step_fn,
            loglikelihood_0=loglikelihood_0,
            **step_parameters,
        )

        def mcmc_kernel(rng_key, state):
            def body_fn(state, rng_key):
                new_state, info = shared_mcmc_step_fn(rng_key, state)
                return new_state, (new_state, info)

            keys = jax.random.split(rng_key, num_mcmc_steps)
            final_state, infos = jax.lax.scan(body_fn, state, keys)
            return final_state, MCMCUpdateInfo(infos[0], infos[1])

        return jax.vmap(mcmc_kernel, in_axes=(0, 0))(rng_key, state)

    return update_function
