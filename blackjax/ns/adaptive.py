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
"""Adaptive Nested Sampling for BlackJAX.

This combines the SMC equivalent of Adaptive Tempering and inner kernel tuning in one file.
"""

from typing import Callable, Dict, NamedTuple, Optional

import jax.numpy as jnp

from blackjax.ns.base import NSInfo, NSState, StateWithLogLikelihood
from blackjax.ns.base import build_kernel as base_build_kernel
from blackjax.ns.base import init as base_init
from blackjax.ns.integrator import NSIntegrator, init_integrator, update_integrator
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["init", "build_kernel"]


class AdaptiveNSState(NamedTuple):
    """An extension of the base NSState to include inner kernel parameters.

    This state class extends the base Nested Sampling state by adding a
    dictionary of parameters for the inner kernel. These parameters can be
    updated adaptively at each step of the Nested Sampling algorithm.

    Attributes
    ----------
    inner_kernel_params
        A dictionary of parameters for the inner kernel used to generate new
        particles during the Nested Sampling process.
    """

    particles: StateWithLogLikelihood
    integrator: NSIntegrator
    inner_kernel_params: Dict[str, ArrayTree]


def init(
    positions: ArrayLikeTree,
    init_state_fn: Callable,
    loglikelihood_birth: Array = jnp.nan,
    update_inner_kernel_params_fn: Optional[Callable] = None,
) -> AdaptiveNSState:
    base_state = base_init(
        positions, init_state_fn, loglikelihood_birth=loglikelihood_birth
    )
    integrator = init_integrator(base_state.particles)
    inner_kernel_params = {}
    if update_inner_kernel_params_fn is not None:
        inner_kernel_params = update_inner_kernel_params_fn(base_state, None, {})
    return AdaptiveNSState(
        particles=base_state.particles,
        inner_kernel_params=inner_kernel_params,
        integrator=integrator,
    )


def build_kernel(
    delete_fn: Callable,
    inner_kernel: Callable,
    update_inner_kernel_params_fn: Callable[
        [NSState, NSInfo, Dict[str, ArrayTree]], Dict[str, ArrayTree]
    ],
) -> Callable:
    """Build an adaptive Nested Sampling kernel."""

    base_kernel = base_build_kernel(
        delete_fn,
        inner_kernel,
    )

    def kernel(
        rng_key: PRNGKey, state: AdaptiveNSState
    ) -> tuple[AdaptiveNSState, NSInfo]:
        """Performs one step of adaptive Nested Sampling."""
        new_state, info = base_kernel(
            rng_key, state, inner_kernel_params=state.inner_kernel_params
        )

        new_inner_kernel_params = update_inner_kernel_params_fn(
            new_state, info, new_state.inner_kernel_params
        )
        new_integrator_state = update_integrator(
            state.integrator, new_state.particles, info.particles
        )
        return (
            AdaptiveNSState(
                particles=new_state.particles,
                inner_kernel_params=new_inner_kernel_params,
                integrator=new_integrator_state,
            ),
            info,
        )

    return kernel
