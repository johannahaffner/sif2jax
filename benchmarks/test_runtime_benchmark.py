# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
import jax
import jax.numpy as jnp
import numpy as np
import pycutest
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
    """Return the most specific Abstract* base class name for grouping.

    Matches the grouping in sif2jax/cutest/__init__.py, e.g.
    AbstractBoundedQuadraticProblem rather than AbstractBoundedMinimisation.
    """
    print(type(problem).__name__)
    for cls in type(problem).__mro__:
        if cls.__name__.startswith("Abstract"):
            return cls.__name__
    return type(problem).__name__


def _test_id(problem):
    """Test ID includes class for -k filtering."""
    return f"{_problem_class(problem)}-{problem.name}"


def _extra_info(problem, implementation):
    return {
        "problem_name": problem.name,
        "dimensionality": problem.y0.size if problem.y0 is not None else 0,
        "has_constraints": hasattr(problem, "constraint"),
        "problem_type": _problem_class(problem),
        "implementation": implementation,
    }


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
    benchmark.extra_info.update(_extra_info(problem, "sif2jax"))
    jax.clear_caches()


@pytest.mark.parametrize("problem", SCALAR_OBJECTIVE_PROBLEMS, ids=_test_id)
def test_sif2jax_val_and_grad_benchmark(benchmark, problem):
    """Benchmark sif2jax value_and_grad for scalar-objective problems."""
    benchmark.group = f"sif2jax-val_and_grad-{_problem_class(problem)}"
    benchmark.name = f"test_sif2jax_val_and_grad_benchmark[{problem.name}]"

    compiled = (
        jax.jit(jax.value_and_grad(problem.objective))
        .lower(problem.y0, problem.args)
        .compile()
    )
    jax.block_until_ready(compiled(problem.y0, problem.args))

    def jax_val_and_grad(y0, args):
        return jax.block_until_ready(compiled(y0, args))

    benchmark(
        jax_val_and_grad, jax.device_put(problem.y0), jax.device_put(problem.args)
    )
    benchmark.extra_info.update(_extra_info(problem, "sif2jax"))
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

    y0 = problem.y0
    v = jnp.ones_like(y0)
    compiled = hvp_fn.lower(y0, v).compile()
    jax.block_until_ready(compiled(y0, v))

    def jax_hvp(y0, v):
        return jax.block_until_ready(compiled(y0, v))

    benchmark(jax_hvp, jax.device_put(y0), jax.device_put(v))
    benchmark.extra_info.update(_extra_info(problem, "sif2jax"))
    jax.clear_caches()


@pytest.mark.parametrize("problem", HESSIAN_PROBLEMS, ids=_test_id)
def test_sif2jax_hessian_benchmark(benchmark, problem):
    """Benchmark sif2jax full Hessian for small problems (n <= 500)."""
    benchmark.group = f"sif2jax-hessian-{_problem_class(problem)}"
    benchmark.name = f"test_sif2jax_hessian_benchmark[{problem.name}]"

    compiled = (
        jax.jit(jax.hessian(problem.objective))
        .lower(problem.y0, problem.args)
        .compile()
    )
    jax.block_until_ready(compiled(problem.y0, problem.args))

    def jax_hessian(y0, args):
        return jax.block_until_ready(compiled(y0, args))

    benchmark(jax_hessian, jax.device_put(problem.y0), jax.device_put(problem.args))
    benchmark.extra_info.update(_extra_info(problem, "sif2jax"))
    jax.clear_caches()


# ---------------------------------------------------------------------------
# pycutest benchmarks (Fortran baseline)
#
# Structured as a class so the pycutest problem fixture uses scope="class" —
# the Fortran shared library is imported once per problem, shared across all
# transform benchmarks (objective, val_and_grad, hvp, hessian), then cleared.
# This prevents accumulating hundreds of dlopen'd .so files in memory.
# ---------------------------------------------------------------------------


def _pycutest_fixture(cls):
    """Add a class-scoped pycutest fixture to a benchmark class."""

    @pytest.fixture(autouse=True, scope="class")
    def pycutest_setup(self, problem):
        try:
            self.__class__._pyc = pycutest.import_problem(
                problem.name, drop_fixed_variables=False
            )
            self.__class__._y0_np = np.asarray(problem.y0, dtype=np.float64)
        except Exception:
            self.__class__._pyc = None
            self.__class__._y0_np = None
        yield
        if self.__class__._pyc is not None:
            pycutest.clear_cache(problem.name)

    cls.pycutest_setup = pycutest_setup
    return cls


@_pycutest_fixture
@pytest.mark.parametrize("problem", ALL_PROBLEMS, ids=_test_id, scope="class")
class TestPycutestObjective:
    """Pycutest objective benchmark — all problems."""

    def test_pycutest_objective_benchmark(self, benchmark, problem):
        if self._pyc is None:
            pytest.skip(f"Could not load {problem.name}")
        benchmark.group = f"pycutest-objective-{_problem_class(problem)}"
        benchmark.name = f"test_pycutest_objective_benchmark[{problem.name}]"
        pyc, y0 = self._pyc, self._y0_np
        _ = pyc.obj(y0)

        benchmark(lambda y: pyc.obj(y), y0)
        benchmark.extra_info.update(_extra_info(problem, "pycutest"))


@_pycutest_fixture
@pytest.mark.parametrize(
    "problem", SCALAR_OBJECTIVE_PROBLEMS, ids=_test_id, scope="class"
)
class TestPycutestDerivatives:
    """Pycutest derivative benchmarks — scalar-objective problems only."""

    def test_pycutest_val_and_grad_benchmark(self, benchmark, problem):
        if self._pyc is None:
            pytest.skip(f"Could not load {problem.name}")
        benchmark.group = f"pycutest-val_and_grad-{_problem_class(problem)}"
        benchmark.name = f"test_pycutest_val_and_grad_benchmark[{problem.name}]"
        pyc, y0 = self._pyc, self._y0_np
        _ = pyc.obj(y0, gradient=True)

        benchmark(lambda y: pyc.obj(y, gradient=True), y0)
        benchmark.extra_info.update(_extra_info(problem, "pycutest"))

    def test_pycutest_hvp_benchmark(self, benchmark, problem):
        if self._pyc is None:
            pytest.skip(f"Could not load {problem.name}")
        benchmark.group = f"pycutest-hvp-{_problem_class(problem)}"
        benchmark.name = f"test_pycutest_hvp_benchmark[{problem.name}]"
        pyc, y0 = self._pyc, self._y0_np
        v = np.ones_like(y0)
        _ = pyc.hprod(y0, v)

        benchmark(lambda y, v: pyc.hprod(y, v), y0, v)
        benchmark.extra_info.update(_extra_info(problem, "pycutest"))


@_pycutest_fixture
@pytest.mark.parametrize("problem", HESSIAN_PROBLEMS, ids=_test_id, scope="class")
class TestPycutestHessian:
    """Pycutest Hessian benchmark — small problems only (n <= 500)."""

    def test_pycutest_hessian_benchmark(self, benchmark, problem):
        if self._pyc is None:
            pytest.skip(f"Could not load {problem.name}")
        benchmark.group = f"pycutest-hessian-{_problem_class(problem)}"
        benchmark.name = f"test_pycutest_hessian_benchmark[{problem.name}]"
        pyc, y0 = self._pyc, self._y0_np
        _ = pyc.hess(y0)

        benchmark(lambda y: pyc.hess(y), y0)
        benchmark.extra_info.update(_extra_info(problem, "pycutest"))


# TODO: Add constraint evaluation and Jacobian benchmarks for
# constrained problems (AbstractConstrainedMinimisation,
# AbstractConstrainedQuadraticProblem) and nonlinear equations
# (AbstractNonlinearEquations). These need constraint(y) and
# jac(constraint)(y) rather than grad/hvp/hessian of the objective.
