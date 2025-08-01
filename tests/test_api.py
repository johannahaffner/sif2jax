import pytest
import sif2jax


def test_get_problem():
    """All problems that are made available through sif2jax.problems should also be
    retrievable by name through the dictionary defined in sif2jax.cutest/__init__.py.
    """
    problem_names = [p.name for p in sif2jax.problems]
    try:
        for name in problem_names:
            sif2jax.cutest.get_problem(name)
    except KeyError as e:
        pytest.fail(f"Problem not found: {e}")


def test_get_problem2():
    """All problems that are made available through sif2jax.cutest.get_problem (which)
    retrieves problems from sif2jax.cutest.problems_dict) should also be available
    through sif2jax.problems.
    """
    problems_in_dict = set(sif2jax.cutest.problems_dict.keys())
    problems_in_sif2jax = set([p.name for p in sif2jax.problems])
    assert problems_in_dict.difference(problems_in_sif2jax) == set()


@pytest.mark.parametrize("problem", sif2jax.quadratic_problems)
def test_quadratic_category(problem):
    assert isinstance(problem, sif2jax.AbstractConstrainedQuadraticProblem)
    assert isinstance(problem, sif2jax.AbstractConstrainedMinimisation)


@pytest.mark.parametrize("problem", sif2jax.unconstrained_minimisation_problems)
def test_unconstrained_category(problem):
    assert isinstance(problem, sif2jax.AbstractUnconstrainedMinimisation)


@pytest.mark.parametrize("problem", sif2jax.bounded_minimisation_problems)
def test_bounded_category(problem):
    assert isinstance(problem, sif2jax.AbstractBoundedMinimisation)


@pytest.mark.parametrize("problem", sif2jax.constrained_minimisation_problems)
def test_constrained_category(problem):
    assert isinstance(problem, sif2jax.AbstractConstrainedMinimisation)


@pytest.mark.parametrize("problem", sif2jax.nonlinear_equations_problems)
def test_nonlinear_equations_category(problem):
    assert isinstance(problem, sif2jax.AbstractNonlinearEquations)
