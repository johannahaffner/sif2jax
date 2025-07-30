"""Runtime benchmarks comparing JAX implementations against pycutest."""

import timeit
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pycutest  # pyright: ignore[reportMissingImports]  - test runs in container
import pytest  # pyright: ignore[reportMissingImports]  - test runs in container
import sif2jax


# Default runtime ratio threshold (JAX time / pycutest time)
DEFAULT_THRESHOLD = 5.0

# Minimum runtime threshold in seconds
# Below this threshold, noise dominates and comparisons are unreliable
MIN_RUNTIME_SECONDS = 1e-5  # 10 microseconds


# pytest_generate_tests is now handled in conftest.py


def benchmark_pycutest(
    func: Callable, *args, number: int = 100, repeat: int = 5
) -> float:
    """Benchmark a pycutest function using timeit.

    Args:
        func: Function to benchmark
        *args: Arguments to pass to func
        number: Number of executions per timing (default: 100)
        repeat: Number of timing repetitions (default: 5)

    Returns:
        Best average time per execution in seconds
    """
    # Warmup run
    _ = func(*args)

    # Create a timer
    timer = timeit.Timer(lambda: func(*args))

    # Run the benchmark
    times = timer.repeat(repeat=repeat, number=number)

    # Return the best average time
    return min(times) / number


def benchmark_jax(
    func: Callable, args: tuple, number: int = 100, repeat: int = 5
) -> float:
    """Benchmark a JAX function using AOT compilation and timeit.

    Args:
        func: JAX function to benchmark (should already be jitted)
        args: Arguments to pass to func as a tuple
        number: Number of executions per timing (default: 100)
        repeat: Number of timing repetitions (default: 5)

    Returns:
        Best average time per execution in seconds
    """
    # AOT compile the function
    compiled = func.lower(*args).compile()

    # Warmup run with block_until_ready
    out = compiled(*args)
    jtu.tree_map(lambda x: x.block_until_ready(), out)

    # Create a timer that calls block_until_ready
    def timed_call():
        out = compiled(*args)  # Map for constraint method: (equalities, inequalities)
        return jtu.tree_map(lambda x: x.block_until_ready(), out)

    timer = timeit.Timer(timed_call)

    # Run the benchmark
    times = timer.repeat(repeat=repeat, number=number)

    # Return the best average time
    return min(times) / number


class TestRuntime:
    """Runtime benchmarks comparing JAX to pycutest."""

    @pytest.fixture(scope="class")
    def pycutest_problem(self, problem):
        """Load pycutest problem once per problem per class."""
        return pycutest.import_problem(problem.name, drop_fixed_variables=False)

    @pytest.fixture(autouse=True)
    def clear_jax_cache(self):
        """Clear JAX cache before each test to ensure fair comparison."""
        jax.clear_caches()
        yield
        jax.clear_caches()

    @pytest.fixture
    def threshold(self, request):
        """Get runtime ratio threshold from command line or use default."""
        threshold_value = request.config.getoption("--runtime-threshold", default=None)
        return float(threshold_value) if threshold_value else DEFAULT_THRESHOLD

    def test_objective_runtime(self, problem, pycutest_problem, threshold):
        """Compare objective function runtime."""
        # Get starting point
        x0 = problem.y0
        args = problem.args

        # Compile JAX function
        jax_obj = jax.jit(problem.objective)

        # Benchmark pycutest
        pycutest_time = benchmark_pycutest(pycutest_problem.obj, x0)

        # Benchmark JAX
        jax_time = benchmark_jax(jax_obj, (x0, args))

        # Calculate ratio
        ratio = jax_time / pycutest_time if pycutest_time > 0 else float("inf")

        # Print results (visible with -s flag)
        print(f"\nObjective runtime for {problem.name}:")
        print(f"  pycutest: {pycutest_time * 1000:.3f} ms")
        print(f"  JAX:      {jax_time * 1000:.3f} ms")
        print(f"  Ratio:    {ratio:.2f}x")

        # Assert within threshold only if runtime is above minimum
        # This avoids failures due to noise in microsecond-level measurements
        if pycutest_time > MIN_RUNTIME_SECONDS:
            assert ratio < threshold, (
                f"JAX objective is {ratio:.2f}x slower than pycutest "
                f"(threshold: {threshold})"
            )

    def test_gradient_runtime(self, problem, pycutest_problem, threshold):
        """Compare gradient computation runtime."""
        # Get starting point
        x0 = problem.y0
        args = problem.args

        # Compile JAX gradient
        jax_grad = jax.jit(jax.grad(problem.objective))

        # Benchmark pycutest
        pycutest_time = benchmark_pycutest(pycutest_problem.grad, x0)

        # Benchmark JAX
        jax_time = benchmark_jax(jax_grad, (x0, args))

        # Calculate ratio
        ratio = jax_time / pycutest_time if pycutest_time > 0 else float("inf")

        # Print results (visible with -s flag)
        print(f"\nGradient runtime for {problem.name}:")
        print(f"  pycutest: {pycutest_time * 1000:.3f} ms")
        print(f"  JAX:      {jax_time * 1000:.3f} ms")
        print(f"  Ratio:    {ratio:.2f}x")

        # Assert within threshold only if runtime is above minimum
        # This avoids failures due to noise in microsecond-level measurements
        if pycutest_time > MIN_RUNTIME_SECONDS:
            assert ratio < threshold, (
                f"JAX gradient is {ratio:.2f}x slower than pycutest "
                f"(threshold: {threshold})"
            )

    def test_constraint_runtime(self, problem, pycutest_problem, threshold):
        """Compare constraint function runtime."""
        # Skip if problem is unconstrained
        if not isinstance(problem, sif2jax.AbstractConstrainedMinimisation):
            pytest.skip("Problem has no constraints")

        # Get starting point
        x0 = problem.y0

        # Compile JAX constraint function
        jax_cons = jax.jit(problem.constraint)

        # Benchmark pycutest
        pycutest_time = benchmark_pycutest(pycutest_problem.cons, x0)

        # Benchmark JAX
        jax_time = benchmark_jax(jax_cons, (x0,))

        # Calculate ratio
        ratio = jax_time / pycutest_time if pycutest_time > 0 else float("inf")

        # Print results (visible with -s flag)
        print(f"\nConstraint runtime for {problem.name}:")
        print(f"  pycutest: {pycutest_time * 1000:.3f} ms")
        print(f"  JAX:      {jax_time * 1000:.3f} ms")
        print(f"  Ratio:    {ratio:.2f}x")

        # Assert within threshold only if runtime is above minimum
        # This avoids failures due to noise in microsecond-level measurements
        if pycutest_time > MIN_RUNTIME_SECONDS:
            assert ratio < threshold, (
                f"JAX constraint is {ratio:.2f}x slower than pycutest "
                f"(threshold: {threshold})"
            )

    def test_constraint_jacobian_runtime(self, problem, pycutest_problem, threshold):
        """Compare constraint Jacobian computation runtime."""
        # Skip if problem is unconstrained
        if not isinstance(problem, sif2jax.AbstractConstrainedMinimisation):
            pytest.skip("Problem has no constraints")

        # Get starting point
        x0 = problem.y0

        # Create JAX Jacobian function
        # For constrained problems, we want the Jacobian of all constraints
        def constraint_wrapper(x):
            eq_cons, ineq_cons = problem.constraint(x)
            # Concatenate all constraints
            all_cons = []
            if eq_cons is not None:
                all_cons.append(eq_cons)
            if ineq_cons is not None:
                all_cons.append(ineq_cons)
            return jnp.concatenate(all_cons) if all_cons else jnp.array([])

        jax_jac = jax.jit(jax.jacfwd(constraint_wrapper))

        # Benchmark pycutest - use dense Jacobian
        # For constrained problems, cjac returns (gradient, Jacobian)
        if hasattr(pycutest_problem, "cjac"):

            def pycutest_jac_func(x):
                _, J = pycutest_problem.cjac(x)
                return J
        else:
            # Skip if no Jacobian method available
            pytest.skip("No Jacobian method available in pycutest")

        pycutest_time = benchmark_pycutest(pycutest_jac_func, x0)

        # Benchmark JAX
        jax_time = benchmark_jax(jax_jac, (x0,))

        # Calculate ratio
        ratio = jax_time / pycutest_time if pycutest_time > 0 else float("inf")

        # Print results (visible with -s flag)
        print(f"\nConstraint Jacobian runtime for {problem.name}:")
        print(f"  pycutest: {pycutest_time * 1000:.3f} ms")
        print(f"  JAX:      {jax_time * 1000:.3f} ms")
        print(f"  Ratio:    {ratio:.2f}x")

        # Assert within threshold only if runtime is above minimum
        # This avoids failures due to noise in microsecond-level measurements
        if pycutest_time > MIN_RUNTIME_SECONDS:
            assert ratio < threshold, (
                f"JAX constraint Jacobian is {ratio:.2f}x slower than pycutest "
                f"(threshold: {threshold})"
            )
