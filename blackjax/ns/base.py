# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base components for Nested Sampling algorithms in BlackJAX.

This module provides the fundamental data structures (`NSState`, `NSInfo`) and
a basic, non-adaptive kernel for Nested Sampling. Nested Sampling is a
Monte Carlo method primarily aimed at Bayesian evidence (marginal likelihood)
computation and posterior sampling, particularly effective for multi-modal
distributions.

The core idea is to transform the multi-dimensional evidence integral into a
one-dimensional integral over the prior volume, ordered by likelihood. This is
achieved by iteratively replacing the point with the lowest likelihood among a
set of "live" points with a new point sampled from the prior, subject to the
constraint that its likelihood must be higher than the one just discarded.

This base implementation uses a provided kernel to perform the constrained
sampling.
"""

from typing import Callable, Dict, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import Array, ArrayLikeTree, PRNGKey

__all__ = ["init", "build_kernel", "NSState", "NSInfo", "delete_fn"]


class StateWithLogLikelihood(NamedTuple):
    position: ArrayLikeTree
    logdensity: Array
    loglikelihood: Array
    loglikelihood_birth: Array


class NSState(NamedTuple):
    """State of the Nested Sampler.

    Contains the current live particles with their positions, log-prior density,
    log-likelihood values, and birth likelihood contours.

    Attributes
    ----------
    position
        A PyTree representing the position of the particles in parameter space.
        The leading dimension corresponds to the number of particles.
    logdensity
        The log-prior density values of the particles.
    loglikelihood
        The log-likelihood values of the particles.
    loglikelihood_birth
        The log-likelihood thresholds that the particles were required to exceed
        when they were sampled (i.e., their birth likelihood contours).
    """

    particles: StateWithLogLikelihood


class NSInfo(NamedTuple):
    """Additional information returned at each step of the Nested Sampling algorithm.

    Attributes
    ----------
    particles
        The NSState of particles that were marked as "dead" (replaced).
        Contains position, logdensity, loglikelihood, and loglikelihood_birth.
    update_info
        A NamedTuple (or any PyTree) containing information from the update step
        (inner kernel) used to generate new live particles.
    """

    particles: NSState
    update_info: NamedTuple


def init_state_strategy(
    position: ArrayLikeTree,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    loglikelihood_birth: Array = jnp.nan,
) -> StateWithLogLikelihood:
    """The default initialisation strategy for each state.

    Parameters
    ----------
    position
        A PyTree of arrays representing the initial positions of the particles.
        Each leaf array has a leading dimension corresponding to the number of particles.
    logprior
        A function that computes the log-prior density for a single particle.
    loglikelihood
        A function that computes the log-likelihood for a single particle.
    loglikelihood_birth
        The log-likelihood threshold that the particle must exceed. Defaults to NaN.

    Returns
    -------
    NSState
        The initialized state containing positions, log-prior, log-likelihood, and birth likelihood.
    """
    logprior_values = logprior_fn(position)
    loglikelihood_values = loglikelihood_fn(position)
    loglikelihood_birth_values = loglikelihood_birth * jnp.ones_like(
        loglikelihood_values
    )

    return StateWithLogLikelihood(
        position, logprior_values, loglikelihood_values, loglikelihood_birth_values
    )


def init(
    positions: ArrayLikeTree,
    init_state_fn: Callable,
    loglikelihood_birth: Array = jnp.nan,
) -> NSState:
    """Initializes the Nested Sampler state.

    Parameters
    ----------
    positions
        An initial set of positions (PyTree of arrays) drawn from the prior
        distribution. The leading dimension of each leaf array must be equal to
        the number of positions.
    init_state_fn
        A function that initializes an NSState from positions.
    loglikelihood_birth
        The initial log-likelihood birth threshold. Defaults to NaN, which
        implies no initial likelihood constraint beyond the prior.

    Returns
    -------
    NSState
        The initial state of the Nested Sampler.
    """
    state_init = init_state_fn(positions)
    loglikelihood_birth_array = loglikelihood_birth * jnp.ones_like(
        state_init.loglikelihood_birth
    )
    return NSState(state_init._replace(loglikelihood_birth=loglikelihood_birth_array))


def build_kernel(
    delete_fn: Callable,
    inner_kernel: Callable,
) -> Callable:
    """Build a generic Nested Sampling kernel.

    This kernel implements one step of the Nested Sampling algorithm. In each step:
    1. A set of particles with the lowest log-likelihoods are identified and
       marked as "dead" using `delete_fn`. The log-likelihood of the "worst"
       of these dead particles (i.e., max among the lowest ones) defines the new
       likelihood constraint `loglikelihood_0`.
    2. Live particles are selected (typically with replacement from the remaining
       live particles, determined by `delete_fn`) to act as starting points for
       the updates.
    3. These selected live particles are evolved using an kernel
       `inner_kernel`. The sampling is constrained to the region where
       `loglikelihood(new_particle) > loglikelihood_0`.
    4. The newly generated particles replace particles marked for replacement,
       (typically the ones that have just been deleted).
    5. The prior volume `logX` and evidence `logZ` are updated based on the
       number of deleted particles and their likelihoods.

    Parameters
    ----------
    delete_fn
        this particle deletion function has the signature
        `(rng_key, current_state) -> (dead_idx, target_update_idx, start_idx)`
        and identifies particles to be deleted, particles to be updated, and
        selects live particles to be starting points for the inner kernel
        for new particle generation.
    inner_kernel
        This kernel function has the signature
        `(rng_keys, inner_state, loglikelihood_0, params) -> (new_inner_state, inner_info)`,
        and is used to generate new particles.

    Returns
    -------
    Callable
        A kernel function for Nested Sampling:
        `(rng_key, state, inner_kernel_params) -> (new_state, ns_info)`.
    """

    def kernel(
        rng_key: PRNGKey, state: NSState, inner_kernel_params: Dict
    ) -> tuple[NSState, NSInfo]:
        # Delete, and grab all the dead information
        rng_key, delete_fn_key = jax.random.split(rng_key)
        dead_idx, target_update_idx, start_idx = delete_fn(
            delete_fn_key, state.particles
        )
        dead_particles = jax.tree.map(lambda x: x[dead_idx], state.particles)

        # Resample the live particles
        sample_keys = jax.random.split(rng_key, len(start_idx))
        inner_state = jax.tree.map(lambda x: x[start_idx], state.particles)
        loglikelihood_0 = dead_particles.loglikelihood.max()
        new_inner_state, inner_update_info = inner_kernel(
            sample_keys, inner_state, loglikelihood_0, inner_kernel_params
        )

        # Update the particles
        state = state._replace(
            particles=jax.tree_util.tree_map(
                lambda p, n: p.at[target_update_idx].set(n),
                state.particles,
                new_inner_state,
            )
        )

        # Return updated state and info
        info = NSInfo(
            dead_particles,
            inner_update_info,
        )
        return state, info

    return kernel


def delete_fn(
    rng_key: PRNGKey, state: StateWithLogLikelihood, num_delete: int
) -> tuple[Array, Array, Array]:
    """Identifies particles to be deleted and selects live particles for resampling.

    This function implements a common strategy in Nested Sampling:
    1. Identify the `num_delete` particles with the lowest log-likelihoods. These
       are marked as "dead".
    2. From the remaining live particles (those not marked as dead), `num_delete`
       particles are chosen (typically with replacement, weighted by their
       current importance weights, here it is uniform from survivors)
       to serve as starting points for generating new particles.

    Parameters
    ----------
    rng_key
        A JAX PRNG key, used here for choosing live particles.
    state
        The current state of the Nested Sampler.
    num_delete
        The number of particles to delete and subsequently replace.

    Returns
    -------
    tuple[Array, Array, Array]
        A tuple containing:
        - `dead_idx`: An array of indices corresponding to the particles
          marked for deletion.
        - `target_update_idx`: An array of indices corresponding to the
          particles to be updated (same as dead_idx in this implementation).
        - `start_idx`: An array of indices corresponding to the particles
            selected for initialization.
    """
    loglikelihood = state.loglikelihood
    neg_dead_loglikelihood, dead_idx = jax.lax.top_k(-loglikelihood, num_delete)
    constraint_loglikelihood = loglikelihood > -neg_dead_loglikelihood.min()
    weights = jnp.array(constraint_loglikelihood, dtype=loglikelihood.dtype)
    weights = jnp.where(weights.sum() > 0.0, weights, jnp.ones_like(weights))
    start_idx = jax.random.choice(
        rng_key,
        len(weights),
        shape=(num_delete,),
        p=weights / weights.sum(),
        replace=True,
    )
    target_update_idx = dead_idx
    return dead_idx, target_update_idx, start_idx
