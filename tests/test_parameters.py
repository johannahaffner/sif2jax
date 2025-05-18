import pycutest
import pytest
import sif2jax


# Dummy test to get started
def test_rosenbrock():
    problem = pycutest.import_problem("ROSENBR")
    assert problem is not None


@pytest.mark.parametrize("problem", sif2jax.problems)
def test_correct_name(problem):
    pycutest_problem = pycutest.import_problem(problem.name())
    assert pycutest_problem is not None


@pytest.mark.parametrize("problem", sif2jax.problems)
def test_correct_dimension(problem):
    pycutest_problem = pycutest.import_problem(problem.name())
    dimensions = problem.y0().size
    assert dimensions == pycutest_problem.n
