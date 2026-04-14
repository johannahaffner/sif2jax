"""Test that objective functions produce clean jaxprs (no gather/scatter).

Gather operations in the jaxpr indicate array indexing patterns that could be
replaced with slices. These are costly for AD because gather's VJP is
scatter-add (an expensive read-modify-write) whereas slice's VJP is pad
(a trivial memory operation).

Problems with modular/permutation indexing that genuinely cannot be expressed
as slices are listed in GATHER_ALLOWED. These use numpy precomputed indices
so the gather has constant (non-traced) index arrays.
"""

import jax
import pytest
import sif2jax
from sif2jax._problem import (
    AbstractBoundedMinimisation,
    AbstractUnconstrainedMinimisation,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("eager_constant_folding", True)

SCALAR_OBJECTIVE_PROBLEMS = [
    p
    for p in sif2jax.problems
    if isinstance(
        p,
        (AbstractUnconstrainedMinimisation, AbstractBoundedMinimisation),
    )
]

# Problems where gather/scatter is accepted — either because the indexing
# pattern is inherently non-sequential (modular permutations) or because
# the scatter-based formulation outperforms the pad-based alternative.
GATHER_ALLOWED = frozenset({
    # Modular permutation indexing (numpy constant indices)
    "NONCVXUN",
    "NONCVXU2",
    "SPARSINE",
    "CVXBQP1",
    "NCVXBQP1",
    "NCVXBQP2",
    "NCVXBQP3",
})


def _collect_primitives(jaxpr):
    """Recursively collect all primitive names from a jaxpr."""
    prims = set()
    for eqn in jaxpr.eqns:
        prims.add(eqn.primitive.name)
        for subjaxpr in jax.core.jaxprs_in_params(eqn.params):
            prims.update(_collect_primitives(subjaxpr))
    return prims


@pytest.mark.parametrize(
    "prob", SCALAR_OBJECTIVE_PROBLEMS, ids=lambda p: p.name
)
def test_no_gather_in_objective(prob):
    """Check that objective jaxpr contains no gather or scatter operations."""
    y0 = prob.y0
    args = prob.args
    jaxpr = jax.make_jaxpr(prob.objective)(y0, args)
    prims = _collect_primitives(jaxpr.jaxpr)

    gather_ops = {p for p in prims if "gather" in p or "scatter" in p}

    if prob.name in GATHER_ALLOWED:
        pytest.skip(
            f"{prob.name} uses modular permutation indexing "
            f"(gather with constant numpy indices)"
        )

    assert not gather_ops, (
        f"{prob.name} objective contains gather/scatter operations: "
        f"{gather_ops}. Replace jnp.arange indexing with slices."
    )
