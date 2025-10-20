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
"""Nested Slice Sampling (NSS) algorithm.
A specific implementation of Nested Sampling that uses
Hit-and-Run Slice Sampling (HRSS) as the inner MCMC kernel.
"""

from functools import partial
from typing import Callable, Dict, NamedTuple, Optional

import jax
import jax.numpy as jnp

from blackjax import SamplingAlgorithm
from blackjax.mcmc.ss import SliceInfo
from blackjax.mcmc.ss import build_kernel as build_slice_kernel
from blackjax.mcmc.ss import sample_direction_from_covariance
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.adaptive import init
from blackjax.ns.base import NSInfo, NSState
from blackjax.ns.base import delete_fn as default_delete_fn
from blackjax.ns.base import PartitionedState
from blackjax.ns.utils import get_first_row, repeat_kernel
from blackjax.smc.tuning.from_particles import particles_covariance_matrix
from blackjax.types import Array, ArrayLikeTree, ArrayTree

__all__ = [
    "init",
    "as_top_level_api",
    "build_kernel",
]


class SliceStateWithLoglikelihood(NamedTuple):
    """State used internally by nested slice sampling.

    This extends the basic SliceState to track both the log-likelihood value
    needed for the likelihood contour constraint.

    Attributes
    ----------
    position
        The current position in parameter space.
    logdensity
        The log-prior probability at the current position.
    loglikelihood
        The log-likelihood value at the current position.
    """

    position: ArrayLikeTree
    logdensity: float
    loglikelihood: Array


class NSSInfo(NamedTuple):
    """Additional information from the Nested Slice Sampling transition.

    Attributes
    ----------
    position
        The position(s) resulting from the slice sampling step.
    logprior
        The log-prior value(s) at the position(s).
    loglikelihood
        The log-likelihood value(s) at the position(s).
    is_accepted
        Whether the slice sampling proposal was accepted.
    num_steps
        The number of steps taken during the stepping-out phase.
    num_shrink
        The number of shrinking steps taken during the shrinking phase.
    """

    position: ArrayTree
    logprior: ArrayTree
    loglikelihood: ArrayTree
    is_accepted: bool
    num_steps: int
    num_shrink: int


def default_stepper_fn(x: ArrayTree, d: ArrayTree, t: float) -> tuple[ArrayTree, bool]:
    """A simple stepper function that moves from `x` along direction `d` by `t` units.

    Implements the operation: `x_new = x + t * d`.

    Parameters
    ----------
    x
        The starting position (PyTree).
    d
        The direction of movement (PyTree, same structure as `x`).
    t
        The scalar step size or distance along the direction.

    Returns
    -------
    tuple[ArrayTree, bool]
        A tuple containing the new position and whether the step was accepted.
    """
    return jax.tree.map(lambda x, d: x + t * d, x, d), True


def compute_covariance_from_particles(
    state: NSState,
    info: NSInfo,
    inner_kernel_params: Optional[Dict[str, ArrayTree]] = None,
) -> Dict[str, ArrayTree]:
    """Compute empirical covariance from current particles for direction proposal."""
    return {
        "cov": jnp.atleast_2d(particles_covariance_matrix(state.particles)),
        "position": get_first_row(state.particles),
    }


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    stepper_fn: Callable = default_stepper_fn,
    adapt_direction_params_fn: Callable = compute_covariance_from_particles,
    generate_slice_direction_fn: Callable = sample_direction_from_covariance,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> Callable:
    """Builds the Nested Slice Sampling kernel.

    see `as_top_level_api` for parameter descriptions.
    """

    @repeat_kernel(num_inner_steps)
    def inner_kernel(
        rng_key, state, logprior_fn, loglikelihood_fn, loglikelihood_0, params
    ):
        rng_key, prop_key = jax.random.split(rng_key, 2)
        d = generate_slice_direction_fn(prop_key, **params)

        def slice_fn(t) -> tuple[SliceStateWithLoglikelihood, bool]:
            x, step_accepted = stepper_fn(state.position, d, t)
            new_state = SliceStateWithLoglikelihood(
                position=x,
                logdensity=logprior_fn(x),
                loglikelihood=loglikelihood_fn(x),
            )
            in_contour = new_state.loglikelihood > loglikelihood_0
            is_accepted = in_contour & step_accepted
            return new_state, is_accepted

        slice_state = SliceStateWithLoglikelihood(
            position=state.position,
            logdensity=state.logprior,
            loglikelihood=state.loglikelihood,
        )

        slice_kernel = build_slice_kernel(slice_fn, max_steps, max_shrinkage)
        new_slice_state, slice_info = slice_kernel(rng_key, slice_state)

        new_state = PartitionedState(
            position=new_slice_state.position,
            logprior=new_slice_state.logdensity,
            loglikelihood=new_slice_state.loglikelihood,
        )
        info = NSSInfo(
            position=new_slice_state.position,
            logprior=new_slice_state.logdensity,
            loglikelihood=new_slice_state.loglikelihood,
            is_accepted=slice_info.is_accepted,
            num_steps=slice_info.num_steps,
            num_shrink=slice_info.num_shrink
        )
        return new_state, info


    delete_fn = partial(default_delete_fn, num_delete=num_delete)

    update_inner_kernel_params_fn = adapt_direction_params_fn
    kernel = build_adaptive_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_fn,
        jax.vmap(inner_kernel, in_axes=(0, 0, None, None, None, None)),
        update_inner_kernel_params_fn,
    )
    return kernel


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    stepper_fn: Callable = default_stepper_fn,
    adapt_direction_params_fn: Callable = compute_covariance_from_particles,
    generate_slice_direction_fn: Callable = sample_direction_from_covariance,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> SamplingAlgorithm:
    """Creates an adaptive Nested Slice Sampling (NSS) algorithm.

    This function configures a Nested Sampling algorithm that uses Hit-and-Run
    Slice Sampling (HRSS) as its inner kernel. The parameters for the HRSS
    direction proposal (specifically, the covariance matrix) are adaptively tuned
    at each step using `adapt_direction_params_fn`.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    num_inner_steps
        The number of HRSS steps to run for each new particle generation.
        This should be a multiple of the dimension of the parameter space.
    num_delete
        The number of particles to delete and replace at each NS step.
        Defaults to 1.
    stepper_fn
        The stepper function `(x, direction, t) -> (x_new, is_accepted)` for the HRSS kernel.
        Defaults to `default_stepper_fn`.
    adapt_direction_params_fn
        A function `(ns_state, ns_info) -> dict_of_params` that computes/adapts
        the parameters (e.g., covariance matrix) for the slice direction proposal,
        based on the current NS state. Defaults to `compute_covariance_from_particles`.
    generate_slice_direction_fn
        A function `(rng_key, **kwargs) -> direction_pytree` that generates a
        normalized direction for HRSS. Keyword arguments are unpacked from the dict
        returned by `adapt_direction_params_fn`. Defaults to `sample_direction_from_covariance`.
    max_steps
        The maximum number of steps to take when expanding the interval in
        each direction during the stepping-out phase. Defaults to 10.
    max_shrinkage
        The maximum number of shrinking steps to perform to avoid infinite loops.
        Defaults to 100.

    Returns
    -------
    SamplingAlgorithm
        A `SamplingAlgorithm` tuple containing `init` and `step` functions for
        the configured Nested Slice Sampler. The state managed by this
        algorithm is `NSState`.
    """

    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        num_inner_steps,
        num_delete,
        stepper_fn=stepper_fn,
        adapt_direction_params_fn=adapt_direction_params_fn,
        generate_slice_direction_fn=generate_slice_direction_fn,
        max_steps=max_steps,
        max_shrinkage=max_shrinkage,
    )

    def init_fn(position, rng_key=None):
        # Vectorize the functions for parallel evaluation over particles
        return init(
            position,
            logprior_fn=jax.vmap(logprior_fn),
            loglikelihood_fn=jax.vmap(loglikelihood_fn),
            update_inner_kernel_params_fn=adapt_direction_params_fn,
        )

    step_fn = kernel

    return SamplingAlgorithm(init_fn, step_fn)
