import jax
import pycutest  # pyright: ignore[reportMissingImports]
import pytest
import sif2jax


# Get all problems for parameterization
ALL_PROBLEMS = list(sif2jax.problems)


@pytest.mark.benchmark(group="sif2jax-objective")
@pytest.mark.parametrize("problem", ALL_PROBLEMS, ids=lambda p: p.name)
def test_sif2jax_objective_benchmark(benchmark, problem):
    """Benchmark sif2jax objective function for all problems.

    This comprehensive benchmark runs on all available problems to document
    performance characteristics for releases.
    """
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
            "problem_type": problem.__class__.__bases__[0].__name__,
            "implementation": "sif2jax",
        }
    )

    # Clear caches after benchmark
    jax.clear_caches()


@pytest.mark.benchmark(group="pycutest-objective")
@pytest.mark.parametrize("problem", ALL_PROBLEMS, ids=lambda p: p.name)
def test_pycutest_objective_benchmark(benchmark, problem):
    """Benchmark pycutest objective function for all problems.

    This provides a comparison baseline against the Fortran implementation.
    """
    # Load pycutest problem
    try:
        pycutest_problem = pycutest.import_problem(
            problem.name, drop_fixed_variables=False
        )
    except Exception as e:
        pytest.skip(f"Could not load pycutest problem {problem.name}: {e}")

    # Warm up
    _ = pycutest_problem.obj(problem.y0)

    # Define function to benchmark
    def pycutest_obj():
        return pycutest_problem.obj(problem.y0)

    # Run benchmark
    benchmark(pycutest_obj)

    # Store extra info for reporting
    benchmark.extra_info.update(
        {
            "problem_name": problem.name,
            "dimensionality": problem.y0.size if problem.y0 is not None else 0,
            "has_constraints": hasattr(problem, "constraint"),
            "problem_type": problem.__class__.__bases__[0].__name__,
            "implementation": "pycutest",
        }
    )

    # Clear pycutest cache
    pycutest.clear_cache(problem.name)


# TODO: Future derivative benchmarking
# Gradient and Hessian benchmarks should be implemented in a problem-class specific way
# since they only make sense for problems with scalar objectives (not for nonlinear
# equation problems, for example). When implemented, they should use benchmark groups
# to organize results by derivative type and problem class.
#
# Example approach for future implementation:
# - Group by problem type (UnconstrainedProblem, BoundedProblem, etc.)
# - Check if problem has scalar objective before computing gradients/Hessians
# - Use separate benchmark groups like "sif2jax-gradient-unconstrained",
#   "pycutest-gradient-unconstrained", etc.
