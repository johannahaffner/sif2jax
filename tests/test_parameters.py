import jax
import jax.numpy as jnp
import numpy as np
import pycutest
import sif2jax


def pytest_generate_tests(metafunc):
    if "problem" in metafunc.fixturenames:
        requested = metafunc.config.getoption("--test-case")
        test_cases = get_test_cases(requested)
        metafunc.parametrize("problem", test_cases)


def get_test_cases(requested):
    if requested is not None:
        try:
            test_case = sif2jax.cutest.get_problem(requested)
        except KeyError:
            raise ValueError(
                f"Test case '{requested}' not found in sif2jax.cutest problems."
            )
        return (test_case,)
    else:
        return sif2jax.problems


def test_correct_name(problem):
    pycutest_problem = pycutest.import_problem(problem.name())
    assert pycutest_problem is not None


def test_correct_dimension(problem):
    pycutest_problem = pycutest.import_problem(problem.name())
    dimensions = problem.y0().size
    assert dimensions == pycutest_problem.n


def test_correct_starting_value(problem):
    pycutest_problem = pycutest.import_problem(problem.name())
    assert np.allclose(pycutest_problem.x0, problem.y0(), rtol=1e-8, atol=1e-8)


def test_correct_objective_at_start(problem):
    pycutest_problem = pycutest.import_problem(problem.name())
    pycutest_value = pycutest_problem.obj(pycutest_problem.x0)
    sif2jax_value = problem.objective(problem.y0(), problem.args())
    assert np.allclose(pycutest_value, sif2jax_value, rtol=1e-6, atol=1e-6)


def test_correct_gradient_at_start(problem):
    pycutest_problem = pycutest.import_problem(problem.name())
    pycutest_gradient = pycutest_problem.grad(pycutest_problem.x0)
    sif2jax_gradient = jax.grad(problem.objective)(problem.y0(), problem.args())
    assert np.allclose(pycutest_gradient, sif2jax_gradient, rtol=1e-8, atol=1e-8)


def test_correct_hessian_at_start(problem):
    pycutest_problem = pycutest.import_problem(problem.name())
    pycutest_hessian = pycutest_problem.ihess(pycutest_problem.x0)
    sif2jax_hessian = jax.hessian(problem.objective)(problem.y0(), problem.args())
    assert np.allclose(pycutest_hessian, sif2jax_hessian, rtol=1e-8, atol=1e-8)


def test_correct_constraints_at_start(problem):
    if isinstance(problem, sif2jax.AbstractConstrainedMinimisation):
        pycutest_problem = pycutest.import_problem(problem.name())
        pycutest_constraints = pycutest_problem.cons(pycutest_problem.x0)
        print(f"pycutest constraints: {pycutest_constraints}")
        # sif2jax_constraints = problem.constraints(problem.y0(), problem.args())
        # assert np.allclose(pycutest_constraints, sif2jax_constraints)
        # TODO choose appropriate tolerance
        pass
    else:
        pass


# test correct constraint jacobian at start


def test_compilation(problem):
    try:
        compiled = jax.jit(problem.objective)
        _ = compiled(problem.y0(), problem.args())
    except Exception as e:
        raise RuntimeError(f"Compilation failed for {problem.name()}") from e


def test_vmap(problem):
    try:
        compiled = jax.vmap(problem.objective, in_axes=(0, None))
        y0 = problem.y0()
        _ = compiled(jnp.array([y0, y0, y0]), problem.args())
    except Exception as e:
        raise RuntimeError(f"Vmap failed for {problem.name}") from e
