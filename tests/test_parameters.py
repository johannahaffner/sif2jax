import jax
import numpy as np
import pycutest
import pytest
import sif2jax


@pytest.mark.parametrize("problem", sif2jax.problems)
def test_correct_name(problem):
    pycutest_problem = pycutest.import_problem(problem.name())
    assert pycutest_problem is not None


@pytest.mark.parametrize("problem", sif2jax.problems)
def test_correct_dimension(problem):
    pycutest_problem = pycutest.import_problem(problem.name())
    dimensions = problem.y0().size
    assert dimensions == pycutest_problem.n


@pytest.mark.parametrize("problem", sif2jax.problems)
def correct_starting_value(problem):
    pycutest_problem = pycutest.import_problem(problem.name())
    assert np.allclose(pycutest_problem.x0, problem.y0(), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("problem", sif2jax.problems)
def test_correct_objective_at_start(problem):
    pycutest_problem = pycutest.import_problem(problem.name())
    pycutest_value = pycutest_problem.obj(pycutest_problem.x0)
    sif2jax_value = problem.objective(problem.y0(), problem.args())
    assert np.allclose(pycutest_value, sif2jax_value, rtol=1e-6, atol=1e-6)


@pytest.mark.skip(reason="Get the objective values right first.")
@pytest.mark.parametrize("problem", sif2jax.problems)
def test_correct_gradient_at_start(problem):
    pycutest_problem = pycutest.import_problem(problem.name())
    pycutest_gradient = pycutest_problem.grad(pycutest_problem.x0)
    sif2jax_gradient = jax.grad(problem.objective)(problem.y0(), problem.args())
    assert np.allclose(pycutest_gradient, sif2jax_gradient, rtol=1e-8, atol=1e-8)


@pytest.mark.skip(reason="Get the objective gradient values right first.")
@pytest.mark.parametrize("problem", sif2jax.problems)
def test_correct_hessian_at_start(problem):
    pycutest_problem = pycutest.import_problem(problem.name())
    pycutest_hessian = pycutest_problem.ihess(pycutest_problem.x0)
    sif2jax_hessian = jax.hessian(problem.objective)(problem.y0(), problem.args())
    assert np.allclose(pycutest_hessian, sif2jax_hessian, rtol=1e-8, atol=1e-8)


