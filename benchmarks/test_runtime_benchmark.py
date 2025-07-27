import jax
import pytest
import sif2jax


# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


def get_problems():
    """Get list of problems lazily to avoid import issues."""
    return list(sif2jax.problems)


def get_problem_ids(problems):
    """Get problem names for test IDs."""
    return [p.name for p in problems]


@pytest.mark.benchmark(group="objective")
def test_objective_benchmark_first(benchmark):
    """Benchmark objective function for the first available problem."""
    problems = get_problems()
    if problems:
        _benchmark_objective(benchmark, problems[0])


@pytest.mark.benchmark(group="objective-subset")
@pytest.mark.parametrize("idx", range(10))
def test_objective_benchmark_subset(benchmark, idx):
    """Benchmark objective function for a subset of problems."""
    problems = get_problems()
    if idx < len(problems):
        _benchmark_objective(benchmark, problems[idx])
    else:
        pytest.skip(f"Problem index {idx} out of range")


def _benchmark_objective(benchmark, problem):
    """Helper function to benchmark a problem's objective function."""
    # Compile JAX function
    jax_objective = jax.jit(problem.objective)
    compiled_objective = jax_objective.lower(problem.y0, problem.args).compile()

    # Warm up
    _ = compiled_objective(problem.y0, problem.args).block_until_ready()

    # Define function to benchmark
    def jax_obj():
        return compiled_objective(problem.y0, problem.args).block_until_ready()

    # Run benchmark
    benchmark(jax_obj)

    # Store extra info for reporting
    benchmark.extra_info.update(
        {
            "problem_name": problem.name,
            "dimensionality": problem.y0.size if problem.y0 is not None else 0,
            "has_constraints": hasattr(problem, "constraint"),
        }
    )

    # Clear caches after benchmark
    jax.clear_caches()


@pytest.mark.benchmark(group="gradient")
def test_gradient_benchmark_first(benchmark):
    """Benchmark gradient computation for the first problem."""
    problems = get_problems()
    if problems:
        problem = problems[0]

        # Create gradient function
        grad_fn = jax.jit(jax.grad(problem.objective, argnums=0))
        compiled_grad = grad_fn.lower(problem.y0, problem.args).compile()

        # Warm up
        _ = compiled_grad(problem.y0, problem.args).block_until_ready()

        # Define function to benchmark
        def jax_grad():
            return compiled_grad(problem.y0, problem.args).block_until_ready()

        # Run benchmark
        benchmark(jax_grad)

        # Store extra info
        benchmark.extra_info.update(
            {
                "problem_name": problem.name,
                "dimensionality": problem.y0.size if problem.y0 is not None else 0,
            }
        )

        # Clear caches
        jax.clear_caches()


@pytest.mark.benchmark(group="hessian")
def test_hessian_benchmark_first(benchmark):
    """Benchmark Hessian computation for small problems."""
    problems = get_problems()

    # Find first small problem
    for problem in problems:
        if problem.y0 is not None and problem.y0.size <= 10:
            # Create Hessian function
            hess_fn = jax.jit(jax.hessian(problem.objective, argnums=0))
            compiled_hess = hess_fn.lower(problem.y0, problem.args).compile()

            # Warm up
            _ = compiled_hess(problem.y0, problem.args).block_until_ready()

            # Define function to benchmark
            def jax_hess():
                return compiled_hess(problem.y0, problem.args).block_until_ready()

            # Run benchmark
            benchmark(jax_hess)

            # Store extra info
            benchmark.extra_info.update(
                {
                    "problem_name": problem.name,
                    "dimensionality": problem.y0.size if problem.y0 is not None else 0,
                }
            )

            # Clear caches
            jax.clear_caches()
            break
    else:
        pytest.skip("No small problems found for Hessian benchmark")
