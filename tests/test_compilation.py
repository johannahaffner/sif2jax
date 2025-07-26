"""Compilation tests to verify implementations don't trigger recompilation."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
import sif2jax


def test_objective_compilation(problem):
    """Test that objective functions don't recompile unnecessarily."""
    y0 = problem.y0
    args = problem.args

    # Wrap objective function for compilation tracking
    @eqx.debug.assert_max_traces(max_traces=1)
    @jax.jit
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
        @eqx.debug.assert_max_traces(max_traces=1)
        @jax.jit
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
    @eqx.debug.assert_max_traces(max_traces=1)
    @jax.jit
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
        @eqx.debug.assert_max_traces(max_traces=1)
        @jax.jit
        def hess_fn(y):
            return jax.hessian(problem.objective)(y, args)

        # Call multiple times - should not recompile
        for _ in range(3):
            hess_fn(y0)

        # Call with different inputs of same shape - should not recompile
        y_test = y0 + 0.1 * jax.random.normal(jax.random.PRNGKey(42), y0.shape)
        hess_fn(y_test)


def test_bounded_inputs_compilation(problem):
    """Test that bounded inputs don't cause recompilation."""
    if hasattr(problem, "bounds") and problem.bounds is not None:
        y0 = problem.y0
        args = problem.args

        # Test objective compilation
        @eqx.debug.assert_max_traces(max_traces=1)
        @jax.jit
        def objective_fn(y):
            return problem.objective(y, args)

        # Call multiple times - should not recompile
        for _ in range(5):
            objective_fn(y0)

        # Test with bounded inputs (project to bounds if needed)
        lower, upper = problem.bounds
        y_bounded = jnp.clip(y0 + 0.1, lower, upper)
        objective_fn(y_bounded)


def test_mixed_input_shapes_trigger_recompilation():
    """Test that different input shapes properly trigger recompilation."""
    try:
        problem = sif2jax.cutest.get_problem("ROSENBR")
    except Exception:
        pytest.skip("ROSENBR problem not available")

    # This should trigger recompilation for each different shape
    call_count = 0

    def counting_objective(y, args):
        nonlocal call_count
        call_count += 1
        return problem.objective(y, args)  # type: ignore

    @jax.jit
    def jit_objective(y):
        return counting_objective(y, problem.args)  # type: ignore

    # Different shapes should trigger recompilation
    y1 = jnp.array([1.0, 2.0])  # Original shape
    y2 = jnp.array([1.0, 2.0, 3.0])  # Different shape

    jit_objective(y1)
    first_count = call_count

    jit_objective(y2)  # Should recompile due to different shape
    second_count = call_count

    # The function should have been called during both compilations
    assert second_count > first_count
