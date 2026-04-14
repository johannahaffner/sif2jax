import jax
import jax.numpy as jnp
import numpy as np
import pycutest  # pyright: ignore[reportMissingImports]
import pytest
import sif2jax
from sif2jax._problem import (
    AbstractBoundedMinimisation,
    AbstractUnconstrainedMinimisation,
)


# Get all problems for parameterization
ALL_PROBLEMS = list(sif2jax.problems)

# Problems with scalar objectives and no general constraints (unconstrained, bounded,
# bounded-quadratic). These are the problems that quasi-Newton solvers like L-BFGS-B can
# handle, and for which grad/hvp/hessian of the objective is meaningful on its own.
SCALAR_OBJECTIVE_PROBLEMS = [
    p
    for p in ALL_PROBLEMS
    if isinstance(p, (AbstractUnconstrainedMinimisation, AbstractBoundedMinimisation))
]

# Hessian benchmarks are expensive — restrict to problems with n <= 500.
HESSIAN_PROBLEMS = [p for p in SCALAR_OBJECTIVE_PROBLEMS if p.y0.size <= 500]


def _problem_class(problem):
    """Return the base problem class name, e.g. 'AbstractUnconstrainedMinimisation'."""
    return type(problem).__bases__[0].__name__


def _test_id(problem):
    """Test ID includes class for -k filtering: 'AbstractUnconstrainedMinimisation-BDQRTIC'."""
    return f"{_problem_class(problem)}-{problem.name}"


# ---------------------------------------------------------------------------
# sif2jax benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("problem", ALL_PROBLEMS, ids=_test_id)
def test_sif2jax_objective_benchmark(benchmark, problem):
    """Benchmark sif2jax objective function for all problems.

    This comprehensive benchmark runs on all available problems to document
    performance characteristics for releases.
    """
    benchmark.group = f"sif2jax-objective-{_problem_class(problem)}"
    benchmark.name = f"test_sif2jax_objective_benchmark[{problem.name}]"

    # Compile JAX function
    jax_objective = jax.jit(problem.objective)
    compiled_objective = jax_objective.lower(problem.y0, problem.args).compile()

    # Warm up
    _ = compiled_objective(problem.y0, problem.args).block_until_ready()

    # Define function to benchmark
    def jax_obj(y0, args):
        return compiled_objective(y0, args).block_until_ready()

    # Run benchmark
    benchmark(jax_obj, jax.device_put(problem.y0), jax.device_put(problem.args))

    # Store extra info for reporting
    benchmark.extra_info.update(
        {
            "problem_name": problem.name,
            "dimensionality": problem.y0.size if problem.y0 is not None else 0,
            "has_constraints": hasattr(problem, "constraint"),
            "problem_type": _problem_class(problem),
            "implementation": "sif2jax",
        }
    )

    # Clear caches after benchmark
    jax.clear_caches()


@pytest.mark.parametrize("problem", SCALAR_OBJECTIVE_PROBLEMS, ids=_test_id)
def test_sif2jax_val_and_grad_benchmark(benchmark, problem):
    """Benchmark sif2jax value_and_grad for scalar-objective problems."""
    benchmark.group = f"sif2jax-val_and_grad-{_problem_class(problem)}"
    benchmark.name = f"test_sif2jax_val_and_grad_benchmark[{problem.name}]"

    # Compile JAX function
    compiled = jax.jit(jax.value_and_grad(problem.objective)).lower(
        problem.y0, problem.args
    ).compile()

    # Warm up
    jax.block_until_ready(compiled(problem.y0, problem.args))

    # Define function to benchmark
    def jax_val_and_grad(y0, args):
        return jax.block_until_ready(compiled(y0, args))

    # Run benchmark
    benchmark(
        jax_val_and_grad, jax.device_put(problem.y0), jax.device_put(problem.args)
    )

    # Store extra info for reporting
    benchmark.extra_info.update(
        {
            "problem_name": problem.name,
            "dimensionality": problem.y0.size if problem.y0 is not None else 0,
            "has_constraints": hasattr(problem, "constraint"),
            "problem_type": _problem_class(problem),
            "implementation": "sif2jax",
        }
    )

    # Clear caches after benchmark
    jax.clear_caches()


@pytest.mark.parametrize("problem", SCALAR_OBJECTIVE_PROBLEMS, ids=_test_id)
def test_sif2jax_hvp_benchmark(benchmark, problem):
    """Benchmark sif2jax Hessian-vector product (forward-over-reverse)."""
    benchmark.group = f"sif2jax-hvp-{_problem_class(problem)}"
    benchmark.name = f"test_sif2jax_hvp_benchmark[{problem.name}]"

    grad_fn = jax.grad(problem.objective, argnums=0)
    args = problem.args

    @jax.jit
    def hvp_fn(y, v):
        _, hv = jax.jvp(lambda y_: grad_fn(y_, args), (y,), (v,))
        return hv

    # Compile JAX function
    y0 = problem.y0
    v = jnp.ones_like(y0)
    compiled = hvp_fn.lower(y0, v).compile()

    # Warm up
    jax.block_until_ready(compiled(y0, v))

    # Define function to benchmark
    def jax_hvp(y0, v):
        return jax.block_until_ready(compiled(y0, v))

    # Run benchmark
    benchmark(jax_hvp, jax.device_put(y0), jax.device_put(v))

    # Store extra info for reporting
    benchmark.extra_info.update(
        {
            "problem_name": problem.name,
            "dimensionality": y0.size if y0 is not None else 0,
            "has_constraints": hasattr(problem, "constraint"),
            "problem_type": _problem_class(problem),
            "implementation": "sif2jax",
        }
    )

    # Clear caches after benchmark
    jax.clear_caches()


@pytest.mark.parametrize("problem", HESSIAN_PROBLEMS, ids=_test_id)
def test_sif2jax_hessian_benchmark(benchmark, problem):
    """Benchmark sif2jax full Hessian for small problems (n <= 500)."""
    benchmark.group = f"sif2jax-hessian-{_problem_class(problem)}"
    benchmark.name = f"test_sif2jax_hessian_benchmark[{problem.name}]"

    # Compile JAX function
    compiled = jax.jit(jax.hessian(problem.objective)).lower(
        problem.y0, problem.args
    ).compile()

    # Warm up
    jax.block_until_ready(compiled(problem.y0, problem.args))

    # Define function to benchmark
    def jax_hessian(y0, args):
        return jax.block_until_ready(compiled(y0, args))

    # Run benchmark
    benchmark(
        jax_hessian, jax.device_put(problem.y0), jax.device_put(problem.args)
    )

    # Store extra info for reporting
    benchmark.extra_info.update(
        {
            "problem_name": problem.name,
            "dimensionality": problem.y0.size if problem.y0 is not None else 0,
            "has_constraints": hasattr(problem, "constraint"),
            "problem_type": _problem_class(problem),
            "implementation": "sif2jax",
        }
    )

    # Clear caches after benchmark
    jax.clear_caches()


# ---------------------------------------------------------------------------
# pycutest benchmarks (Fortran baseline)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("problem", ALL_PROBLEMS, ids=_test_id)
def test_pycutest_objective_benchmark(benchmark, problem):
    """Benchmark pycutest objective function for all problems.

    This provides a comparison baseline against the Fortran implementation.
    """
    benchmark.group = f"pycutest-objective-{_problem_class(problem)}"
    benchmark.name = f"test_pycutest_objective_benchmark[{problem.name}]"

    # Load pycutest problem
    try:
        pycutest_problem = pycutest.import_problem(
            problem.name, drop_fixed_variables=False
        )
    except Exception as e:
        pytest.skip(f"Could not load pycutest problem {problem.name}: {e}")

    # Convert to numpy f64 — pycutest's Fortran module requires numpy arrays
    y0_np = np.asarray(problem.y0, dtype=np.float64)

    # Warm up
    _ = pycutest_problem.obj(y0_np)

    # Define function to benchmark
    def pycutest_obj(y0):
        return pycutest_problem.obj(y0)

    # Run benchmark
    benchmark(pycutest_obj, y0_np)

    # Store extra info for reporting
    benchmark.extra_info.update(
        {
            "problem_name": problem.name,
            "dimensionality": y0_np.size if y0_np is not None else 0,
            "has_constraints": hasattr(problem, "constraint"),
            "problem_type": _problem_class(problem),
            "implementation": "pycutest",
        }
    )

    # Clear pycutest cache
    pycutest.clear_cache(problem.name)


@pytest.mark.parametrize("problem", SCALAR_OBJECTIVE_PROBLEMS, ids=_test_id)
def test_pycutest_val_and_grad_benchmark(benchmark, problem):
    """Benchmark pycutest objective+gradient for scalar-objective problems."""
    benchmark.group = f"pycutest-val_and_grad-{_problem_class(problem)}"
    benchmark.name = f"test_pycutest_val_and_grad_benchmark[{problem.name}]"

    # Load pycutest problem
    try:
        pycutest_problem = pycutest.import_problem(
            problem.name, drop_fixed_variables=False
        )
    except Exception as e:
        pytest.skip(f"Could not load pycutest problem {problem.name}: {e}")

    # Convert to numpy f64 — pycutest's Fortran module requires numpy arrays
    y0_np = np.asarray(problem.y0, dtype=np.float64)

    # Warm up
    _ = pycutest_problem.obj(y0_np, gradient=True)

    # Define function to benchmark
    def pycutest_val_and_grad(y0):
        return pycutest_problem.obj(y0, gradient=True)

    # Run benchmark
    benchmark(pycutest_val_and_grad, y0_np)

    # Store extra info for reporting
    benchmark.extra_info.update(
        {
            "problem_name": problem.name,
            "dimensionality": y0_np.size,
            "has_constraints": hasattr(problem, "constraint"),
            "problem_type": _problem_class(problem),
            "implementation": "pycutest",
        }
    )

    # Clear pycutest cache
    pycutest.clear_cache(problem.name)


@pytest.mark.parametrize("problem", SCALAR_OBJECTIVE_PROBLEMS, ids=_test_id)
def test_pycutest_hvp_benchmark(benchmark, problem):
    """Benchmark pycutest Hessian-vector product."""
    benchmark.group = f"pycutest-hvp-{_problem_class(problem)}"
    benchmark.name = f"test_pycutest_hvp_benchmark[{problem.name}]"

    # Load pycutest problem
    try:
        pycutest_problem = pycutest.import_problem(
            problem.name, drop_fixed_variables=False
        )
    except Exception as e:
        pytest.skip(f"Could not load pycutest problem {problem.name}: {e}")

    # Convert to numpy f64 — pycutest's Fortran module requires numpy arrays
    y0_np = np.asarray(problem.y0, dtype=np.float64)
    v_np = np.ones_like(y0_np)

    # Warm up
    _ = pycutest_problem.hprod(y0_np, v_np)

    # Define function to benchmark
    def pycutest_hvp(y0, v):
        return pycutest_problem.hprod(y0, v)

    # Run benchmark
    benchmark(pycutest_hvp, y0_np, v_np)

    # Store extra info for reporting
    benchmark.extra_info.update(
        {
            "problem_name": problem.name,
            "dimensionality": y0_np.size,
            "has_constraints": hasattr(problem, "constraint"),
            "problem_type": _problem_class(problem),
            "implementation": "pycutest",
        }
    )

    # Clear pycutest cache
    pycutest.clear_cache(problem.name)


@pytest.mark.parametrize("problem", HESSIAN_PROBLEMS, ids=_test_id)
def test_pycutest_hessian_benchmark(benchmark, problem):
    """Benchmark pycutest full Hessian for small problems (n <= 500)."""
    benchmark.group = f"pycutest-hessian-{_problem_class(problem)}"
    benchmark.name = f"test_pycutest_hessian_benchmark[{problem.name}]"

    # Load pycutest problem
    try:
        pycutest_problem = pycutest.import_problem(
            problem.name, drop_fixed_variables=False
        )
    except Exception as e:
        pytest.skip(f"Could not load pycutest problem {problem.name}: {e}")

    # Convert to numpy f64 — pycutest's Fortran module requires numpy arrays
    y0_np = np.asarray(problem.y0, dtype=np.float64)

    # Warm up
    _ = pycutest_problem.hess(y0_np)

    # Define function to benchmark
    def pycutest_hessian(y0):
        return pycutest_problem.hess(y0)

    # Run benchmark
    benchmark(pycutest_hessian, y0_np)

    # Store extra info for reporting
    benchmark.extra_info.update(
        {
            "problem_name": problem.name,
            "dimensionality": y0_np.size,
            "has_constraints": hasattr(problem, "constraint"),
            "problem_type": _problem_class(problem),
            "implementation": "pycutest",
        }
    )

    # Clear pycutest cache
    pycutest.clear_cache(problem.name)


# TODO: Add constraint evaluation and Jacobian benchmarks for constrained problems
# (AbstractConstrainedMinimisation, AbstractConstrainedQuadraticProblem) and nonlinear
# equations (AbstractNonlinearEquations). These need constraint(y) and jac(constraint)(y)
# benchmarks rather than grad/hvp/hessian of the objective.
