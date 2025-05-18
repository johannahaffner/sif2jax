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
