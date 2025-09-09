import timeit
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pycutest  # pyright: ignore[reportMissingImports]  - test runs in container
import pytest  # pyright: ignore[reportMissingImports]  - test runs in container
import sif2jax


DEFAULT_THRESHOLD = 5.0  # Our implementation may at most be five times slower
MIN_RUNTIME_SECONDS = 1e-5  # Noise dominates at the microsecond level


def benchmark_pycutest(
    func: Callable, point, number: int = 100, repeat: int = 5
) -> float:
    """Benchmarking a pycutest function (`func`)."""
    _ = func(np.asarray(point))
    timer = timeit.Timer(lambda: func(np.asarray(point)))
    times = timer.repeat(repeat=repeat, number=number)
    return min(times) / number


def benchmark_jax(
    func: Callable, args: tuple, number: int = 100, repeat: int = 5
) -> float:
    """Benchmark a JAX function using AOT compilation and timeit."""
    compiled = func.lower(*args).compile()
    out = compiled(*args)
    jtu.tree_map(lambda x: x.block_until_ready(), out)

    def timed_call():
        out = compiled(*args)  # Map for constraint method: (equalities, inequalities)
        return jtu.tree_map(lambda x: x.block_until_ready(), out)

    timer = timeit.Timer(timed_call)
    times = timer.repeat(repeat=repeat, number=number)
    return min(times) / number


@pytest.fixture(autouse=True)
def clear_caches():
    jax.clear_caches()


@pytest.fixture(scope="class")
def pycutest_problem(problem):
    pycutest_problem_ = pycutest.import_problem(
        problem.name, drop_fixed_variables=False
    )
    yield pycutest_problem_

    pycutest.clear_cache(problem.name)


@pytest.fixture
def threshold(request):
    """Get runtime ratio threshold from command line or use default."""
    threshold_value = request.config.getoption("--runtime-threshold", default=None)
    return float(threshold_value) if threshold_value else DEFAULT_THRESHOLD


class TestRuntime:
    """Runtime benchmarks comparing JAX to pycutest."""

    def test_objective_runtime(self, problem, pycutest_problem, threshold):
        pycutest_time = benchmark_pycutest(pycutest_problem.obj, problem.y0)

        jax_obj = jax.jit(problem.objective)
        jax_time = benchmark_jax(jax_obj, (problem.y0, problem.args))

        ratio = jax_time / pycutest_time if pycutest_time > 0 else float("inf")

        # Print results (visible with -s flag)
        print(f"\nObjective runtime for {problem.name}:")
        print(f"  pycutest: {pycutest_time * 1000:.3f} ms")
        print(f"  JAX:      {jax_time * 1000:.3f} ms")
        print(f"  Ratio:    {ratio:.2f}x")

        if pycutest_time > MIN_RUNTIME_SECONDS:
            assert ratio < threshold, (
                f"JAX objective is {ratio:.2f}x slower than pycutest "
                f"(threshold: {threshold})"
            )

    def test_gradient_runtime(self, problem, pycutest_problem, threshold):
        """Compare gradient computation runtime."""
        pycutest_time = benchmark_pycutest(pycutest_problem.grad, problem.y0)

        jax_grad = jax.jit(jax.grad(problem.objective))
        jax_time = benchmark_jax(jax_grad, (problem.y0, problem.args))

        ratio = jax_time / pycutest_time if pycutest_time > 0 else float("inf")

        # Print results (visible with -s flag)
        print(f"\nGradient runtime for {problem.name}:")
        print(f"  pycutest: {pycutest_time * 1000:.3f} ms")
        print(f"  JAX:      {jax_time * 1000:.3f} ms")
        print(f"  Ratio:    {ratio:.2f}x")

        if pycutest_time > MIN_RUNTIME_SECONDS:
            assert ratio < threshold, (
                f"JAX gradient is {ratio:.2f}x slower than pycutest "
                f"(threshold: {threshold})"
            )

    def test_constraint_runtime(self, problem, pycutest_problem, threshold):
        """Compare constraint function runtime."""
        if not isinstance(problem, sif2jax.AbstractConstrainedMinimisation):
            pytest.skip("Problem has no constraints")

        pycutest_time = benchmark_pycutest(pycutest_problem.cons, problem.y0)

        jax_cons = jax.jit(problem.constraint)
        jax_time = benchmark_jax(jax_cons, (problem.y0,))

        ratio = jax_time / pycutest_time if pycutest_time > 0 else float("inf")

        # Print results (visible with -s flag)
        print(f"\nConstraint runtime for {problem.name}:")
        print(f"  pycutest: {pycutest_time * 1000:.3f} ms")
        print(f"  JAX:      {jax_time * 1000:.3f} ms")
        print(f"  Ratio:    {ratio:.2f}x")

        if pycutest_time > MIN_RUNTIME_SECONDS:
            assert ratio < threshold, (
                f"JAX constraint is {ratio:.2f}x slower than pycutest "
                f"(threshold: {threshold})"
            )

    def test_constraint_jacobian_runtime(self, problem, pycutest_problem, threshold):
        """Compare constraint Jacobian computation runtime."""
        if not isinstance(problem, sif2jax.AbstractConstrainedMinimisation):
            pytest.skip("Problem has no constraints")

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

        pycutest_time = benchmark_pycutest(pycutest_jac_func, problem.y0)
        jax_time = benchmark_jax(jax_jac, (problem.y0,))

        ratio = jax_time / pycutest_time if pycutest_time > 0 else float("inf")

        # Print results (visible with -s flag)
        print(f"\nConstraint Jacobian runtime for {problem.name}:")
        print(f"  pycutest: {pycutest_time * 1000:.3f} ms")
        print(f"  JAX:      {jax_time * 1000:.3f} ms")
        print(f"  Ratio:    {ratio:.2f}x")

        if pycutest_time > MIN_RUNTIME_SECONDS:
            assert ratio < threshold, (
                f"JAX constraint Jacobian is {ratio:.2f}x slower than pycutest "
                f"(threshold: {threshold})"
            )
