#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for speeding up optimization in tests.

"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager, ExitStack
from functools import wraps
from typing import Any, Callable
from unittest import mock

from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
)
from botorch.optim.utils.timeout import minimize_with_timeout
from scipy.optimize import OptimizeResult
from torch import Tensor


@contextmanager
def mock_optimize_context_manager(
    force: bool = False,
) -> Generator[None, None, None]:
    """A context manager that uses mocks to speed up optimization for testing.
    Currently, the primary tactic is to force the underlying scipy methods to stop
    after just one iteration.

        force: If True will not raise an AssertionError if no mocks are called.
            USE RESPONSIBLY.
    """

    def one_iteration_minimize(*args: Any, **kwargs: Any) -> OptimizeResult:
        if kwargs["options"] is None:
            kwargs["options"] = {}

        kwargs["options"]["maxiter"] = 1
        return minimize_with_timeout(*args, **kwargs)

    def minimal_gen_ics(*args: Any, **kwargs: Any) -> Tensor:
        kwargs["num_restarts"] = 2
        kwargs["raw_samples"] = 4

        return gen_batch_initial_conditions(*args, **kwargs)

    def minimal_gen_os_ics(*args: Any, **kwargs: Any) -> Tensor | None:
        kwargs["num_restarts"] = 2
        kwargs["raw_samples"] = 4

        return gen_one_shot_kg_initial_conditions(*args, **kwargs)

    with ExitStack() as es:
        # Note this `minimize_with_timeout` is defined in optim.utils.timeout;
        # this mock only has an effect when calling a function used in
        # `botorch.generation.gen`, such as `gen_candidates_scipy`.
        mock_generation = es.enter_context(
            mock.patch(
                "botorch.generation.gen.minimize_with_timeout",
                wraps=one_iteration_minimize,
            )
        )

        # Similarly, works when using calling a function defined in
        # `optim.core`, such as `scipy_minimize` and `torch_minimize`.
        mock_fit = es.enter_context(
            mock.patch(
                "botorch.optim.core.minimize_with_timeout",
                wraps=one_iteration_minimize,
            )
        )

        # Works when calling a function in `optim.optimize` such as
        # `optimize_acqf`
        mock_gen_ics = es.enter_context(
            mock.patch(
                "botorch.optim.optimize.gen_batch_initial_conditions",
                wraps=minimal_gen_ics,
            )
        )

        # Works when calling a function in `optim.optimize` such as
        # `optimize_acqf`
        mock_gen_os_ics = es.enter_context(
            mock.patch(
                "botorch.optim.optimize.gen_one_shot_kg_initial_conditions",
                wraps=minimal_gen_os_ics,
            )
        )

        # Reduce default number of iterations in `optimize_acqf_mixed_alternating`.
        for name in [
            "MAX_ITER_ALTER",
            "MAX_ITER_DISCRETE",
            "MAX_ITER_CONT",
        ]:
            es.enter_context(mock.patch(f"botorch.optim.optimize_mixed.{name}", new=1))

        yield

    if (not force) and all(
        mock_.call_count < 1
        for mock_ in [
            mock_generation,
            mock_fit,
            mock_gen_ics,
            mock_gen_os_ics,
        ]
    ):
        raise AssertionError(
            "No mocks were called in the context manager. Please remove unused "
            "mock_optimize_context_manager()."
        )


def mock_optimize(f: Callable) -> Callable:
    """Wraps `f` in `mock_optimize_context_manager` for use as a decorator."""

    @wraps(f)
    # pyre-fixme[3]: Return type must be annotated.
    def inner(*args: Any, **kwargs: Any):
        with mock_optimize_context_manager():
            return f(*args, **kwargs)

    return inner
