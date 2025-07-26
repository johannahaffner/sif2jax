"""Compilation tests to verify implementations don't trigger recompilation."""

import equinox as eqx
import jax


def test_objective_compilation(problem):
    """Test that objective functions don't recompile unnecessarily."""
    y0 = problem.y0
    args = problem.args

    # Wrap objective function for compilation tracking
    @jax.jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def objective_fn(y):
        return problem.objective(y, args)

    # Call with different inputs of same shape - should not recompile
    for _ in range(3):
        y_test = y0 + 0.1 * jax.random.normal(jax.random.PRNGKey(42), y0.shape)
        objective_fn(y_test)


def test_constraint_compilation(problem):
    """Test that constraint functions don't recompile unnecessarily."""
    if hasattr(problem, "constraint"):
        y0 = problem.y0

        # Wrap constraint function for compilation tracking
        @jax.jit
        @eqx.debug.assert_max_traces(max_traces=1)
        def constraint_fn(y):
            return problem.constraint(y)

        # Call with different inputs of same shape - should not recompile
        for _ in range(3):
            y_test = y0 + 0.1 * jax.random.normal(jax.random.PRNGKey(42), y0.shape)
            constraint_fn(y_test)


# TODO: Add test for mixed input shapes triggering recompilation
# This will be needed once we support variably dimensioned problems.
# The test should verify that different input shapes properly trigger
# recompilation, while same shapes don't cause unnecessary recompilation.
