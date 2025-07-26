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

    # Call multiple times with same input - should not recompile
    for _ in range(5):
        objective_fn(y0)

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

        # Call multiple times - should not recompile
        for _ in range(5):
            constraint_fn(y0)

        # Call with different inputs of same shape - should not recompile
        for _ in range(3):
            y_test = y0 + 0.1 * jax.random.normal(jax.random.PRNGKey(42), y0.shape)
            constraint_fn(y_test)


def test_gradient_compilation(problem):
    """Test that gradient computations don't recompile unnecessarily."""
    y0 = problem.y0
    args = problem.args

    # Wrap gradient function for compilation tracking
    @jax.jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def grad_fn(y):
        return jax.grad(problem.objective)(y, args)

    # Call multiple times - should not recompile
    for _ in range(5):
        grad_fn(y0)

    # Call with different inputs of same shape - should not recompile
    for _ in range(3):
        y_test = y0 + 0.1 * jax.random.normal(jax.random.PRNGKey(42), y0.shape)
        grad_fn(y_test)


def test_hessian_compilation(problem):
    """Test that Hessian computations don't recompile unnecessarily."""
    # Only test small problems for Hessian due to computational cost
    if problem.n <= 10:
        y0 = problem.y0
        args = problem.args

        # Wrap Hessian function for compilation tracking
        @jax.jit
        @eqx.debug.assert_max_traces(max_traces=1)
        def hess_fn(y):
            return jax.hessian(problem.objective)(y, args)

        # Call multiple times - should not recompile
        for _ in range(3):
            hess_fn(y0)

        # Call with different inputs of same shape - should not recompile
        y_test = y0 + 0.1 * jax.random.normal(jax.random.PRNGKey(42), y0.shape)
        hess_fn(y_test)


# TODO: Add test for mixed input shapes triggering recompilation
# This will be needed once we support variably dimensioned problems.
# The test should verify that different input shapes properly trigger
# recompilation, while same shapes don't cause unnecessary recompilation.
