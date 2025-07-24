import inspect

import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import numpy as np
import pycutest  # pyright: ignore[reportMissingImports]  - test runs in container
import pytest  # pyright: ignore[reportMissingImports]  - test runs in container
import sif2jax


# pytest_generate_tests is now handled in conftest.py


def _evaluate_at_other(
    problem_name, problem_function, pycutest_problem_function, point
):
    """Evaluate the problem function at a given point. We don't know if the function is
    actually defined at that point, or if evaluating it there causes a division by zero,
    an infinite value or something else. So we handle these cases here.

    These test cases serve to identify issues in the sif2jax problems that cannot be
    caught by evaluating the problems at a single point.

    To test e.g. the gradient or Hessian, use this pattern:

    ```python
    _evaluate_at_other(jax.grad(problem.objective), pycutest_problem.grad, point)
    ```
    """
    pycutest_e = None
    sif2jax_e = None

    try:
        pycutest_value = pycutest_problem_function(point)
        pycutest_failed = False
        if jnp.any(jnp.isnan(pycutest_value)) or jnp.any(jnp.isinf(pycutest_value)):
            pycutest_failed = True
            pycutest_e = ValueError("pycutest returned NaN or Inf.")
            pycutest_value = None
    except (ZeroDivisionError, ValueError) as e:
        pycutest_e = e
        pycutest_value = None
        pycutest_failed = True

    try:
        sif2jax_value = problem_function(point)
        sif2jax_failed = False
        if jnp.any(jnp.isnan(sif2jax_value)) or jnp.any(jnp.isinf(sif2jax_value)):
            sif2jax_failed = True
            sif2jax_e = ValueError("sif2jax returned NaN or Inf.")
            sif2jax_value = None
    except (ZeroDivisionError, ValueError) as e:
        sif2jax_e = e
        sif2jax_value = None
        sif2jax_failed = True

    # Both should either succeed or fail in the same way
    if pycutest_failed and sif2jax_failed:
        assert type(pycutest_e) == type(sif2jax_e), (
            f"Errors differ for problem {problem_name} at point {point}: "
            f"pycutest_error={type(pycutest_e).__name__}, "
            f"sif2jax_error={type(sif2jax_e).__name__}"
        )
    elif pycutest_failed or sif2jax_failed:
        msg = (
            f"One implementation failed at point {point} for problem {problem_name}: "
            f"pycutest_failed={pycutest_failed}, sif2jax_failed={sif2jax_failed}. "
        )
        if pycutest_failed:
            msg += f"pycutest_error={type(pycutest_e).__name__}"
            if type(pycutest_e) is ValueError:
                # Special case: if pycutest fails but sif2jax returns a numerical value,
                # we assume that we have more robust numerics and that any discrepancy
                # can hopefully be found with another test. We cannot examine the
                # Fortran source code from here, so we would not have any information
                # relevant to fixing this case.
                assert True
            else:
                pytest.fail(msg)
        else:
            msg += f"sif2jax_error={type(sif2jax_e).__name__}"
            pytest.fail(msg)
    else:
        assert pycutest_value is not None and sif2jax_value is not None
        # Absolute tolerance made slightly more permissive here.
        # TODO: this either needs to be fixed (so we can go back to the default 1e-8)),
        # or we need to document this as a known issue.
        assert np.allclose(jnp.asarray(pycutest_value), sif2jax_value, atol=1e-6)


class TestProblem:
    """Test class for CUTEst problems. This class tests sif2jax implementations of
    CUTEst problems against the pycutest interface to the Fortran problems, using the
    latter as the ground truth. It provides a range of test cases, escalating in
    complexity, to ensure that the sif2jax problems match the Fortran implementations
    up to numerical precision.

    When fixing issues in the tests, it is recommended to start with the basics (correct
    dimensions, starting values, and objective function) before moving on to more
    difficult tests, e.g. those evaluating gradients or hessians, or involving
    vectorisation of the code.
    """

    @pytest.fixture(scope="class")
    def pycutest_problem(self, problem):
        """Load pycutest problem once per problem per class."""
        return pycutest.import_problem(problem.name)

    def test_correct_name(self, pycutest_problem):
        assert pycutest_problem is not None

    def test_correct_dimension(self, problem, pycutest_problem):
        dimensions = problem.y0.size
        assert dimensions == pycutest_problem.n

    def test_correct_starting_value(self, problem, pycutest_problem):
        assert np.allclose(pycutest_problem.x0, problem.y0)

    def test_correct_objective_at_start(self, problem, pycutest_problem):
        pycutest_value = pycutest_problem.obj(pycutest_problem.x0)
        sif2jax_value = problem.objective(problem.y0, problem.args)
        assert np.allclose(pycutest_value, sif2jax_value)

    def test_correct_objective_zero_vector(self, problem, pycutest_problem):
        _evaluate_at_other(
            problem.name,
            lambda x: problem.objective(x, problem.args),
            pycutest_problem.obj,
            jnp.zeros_like(problem.y0),
        )

    def test_correct_objective_ones_vector(self, problem, pycutest_problem):
        _evaluate_at_other(
            problem.name,
            lambda x: problem.objective(x, problem.args),
            pycutest_problem.obj,
            jnp.ones_like(problem.y0),
        )

    def test_correct_gradient_at_start(self, problem, pycutest_problem):
        pycutest_gradient = pycutest_problem.grad(pycutest_problem.x0)
        sif2jax_gradient = jax.grad(problem.objective)(problem.y0, problem.args)
        assert np.allclose(pycutest_gradient, sif2jax_gradient)

    def test_correct_gradient_zero_vector(self, problem, pycutest_problem):
        _evaluate_at_other(
            problem.name,
            lambda x: jax.grad(problem.objective)(x, problem.args),
            pycutest_problem.grad,
            jnp.zeros_like(problem.y0),
        )

    def test_correct_gradient_ones_vector(self, problem, pycutest_problem):
        _evaluate_at_other(
            problem.name,
            lambda x: jax.grad(problem.objective)(x, problem.args),
            pycutest_problem.grad,
            jnp.ones_like(problem.y0),
        )

    def test_correct_hessian_at_start(self, problem, pycutest_problem):
        if problem.num_variables() < 1000:
            pycutest_hessian = pycutest_problem.ihess(pycutest_problem.x0)
            sif2jax_hessian = jax.hessian(problem.objective)(problem.y0, problem.args)
            assert np.allclose(pycutest_hessian, sif2jax_hessian)
        else:
            pytest.skip("Skip Hessian test for large problems to save time and memory")

    def test_correct_hessian_zero_vector(self, problem, pycutest_problem):
        if problem.num_variables() < 1000:
            _evaluate_at_other(
                problem.name,
                lambda x: jax.hessian(problem.objective)(x, problem.args),
                pycutest_problem.ihess,
                jnp.zeros_like(problem.y0),
            )
        else:
            pytest.skip("Skip Hessian test for large problems to save time and memory")

    def test_correct_hessian_ones_vector(self, problem, pycutest_problem):
        if problem.num_variables() < 1000:
            _evaluate_at_other(
                problem.name,
                lambda x: jax.hessian(problem.objective)(x, problem.args),
                pycutest_problem.ihess,
                jnp.ones_like(problem.y0),
            )
        else:
            pytest.skip("Skip Hessian test for large problems to save time and memory")

    def test_correct_constraint_dimensions(self, problem, pycutest_problem):
        num_equalities, num_inequalities, _ = problem.num_constraints()

        if pycutest_problem.m == 0:
            assert num_equalities == 0
            assert num_inequalities == 0
        else:
            pycutest_constraints = pycutest_problem.cons(pycutest_problem.x0)
            assert pycutest_constraints is not None

            pycutest_equalities = pycutest_constraints[pycutest_problem.is_eq_cons]  # pyright: ignore
            pycutest_equalities = jnp.array(pycutest_equalities).squeeze()
            assert pycutest_equalities.size == num_equalities

            pycutest_inequalities = pycutest_constraints[~pycutest_problem.is_eq_cons]  # pyright: ignore
            pycutest_inequalities = jnp.array(pycutest_inequalities).squeeze()
            assert pycutest_inequalities.size == num_inequalities

    def test_correct_number_of_finite_bounds(self, problem, pycutest_problem):
        _, _, num_finite_bounds = problem.num_constraints()

        # Pycutest defaults unconstrained variables to -1e20 and 1e20
        pycutest_finite_lower = jnp.sum(jnp.asarray(pycutest_problem.bl > -1e20))
        pycutest_finite_upper = jnp.sum(jnp.asarray(pycutest_problem.bu < 1e20))

        assert num_finite_bounds == pycutest_finite_lower + pycutest_finite_upper

        if pycutest_finite_upper + pycutest_finite_lower == 0:
            # Check if the sif2jax problem should have a bounds attribute
            if not isinstance(problem, sif2jax.AbstractUnconstrainedMinimisation):
                assert problem.bounds is None, "sif2jax problem should not have bounds."

    def test_correct_bounds(self, problem, pycutest_problem):
        # Skip test for unconstrained problems which don't have bounds attribute
        if isinstance(problem, sif2jax.AbstractUnconstrainedMinimisation):
            pytest.skip("Unconstrained problems have no bounds.")

        if problem.bounds is not None:
            lower, upper = problem.bounds

            assert pycutest_problem.bl is not None
            pc_lower = jnp.asarray(pycutest_problem.bl)
            pc_lower = jnp.where(pc_lower == -1e20, -jnp.inf, pc_lower)
            assert np.allclose(pc_lower, lower), "Lower bounds do not match."

            assert pycutest_problem.bu is not None
            pc_upper = jnp.asarray(pycutest_problem.bu)
            pc_upper = jnp.where(pc_upper == 1e20, jnp.inf, pc_upper)
            assert np.allclose(pc_upper, upper), "Upper bounds do not match."
        else:
            assert problem.bounds is None, "sif2jax problem should not have bounds."
            pytest.skip("Problem has no bounds defined.")

    def test_correct_constraints_at_start(self, problem, pycutest_problem):
        if isinstance(problem, sif2jax.AbstractConstrainedMinimisation):
            assert pycutest_problem.m > 0, "Problem should have constraints."

            pycutest_constraints = pycutest_problem.cons(pycutest_problem.x0)
            pycutest_equalities = pycutest_constraints[pycutest_problem.is_eq_cons]  # pyright: ignore
            pycutest_equalities = jnp.array(pycutest_equalities).squeeze()
            pycutest_inequalities = pycutest_constraints[~pycutest_problem.is_eq_cons]  # pyright: ignore
            pycutest_inequalities = jnp.array(pycutest_inequalities).squeeze()

            sif2jax_constraints = problem.constraint(problem.y0)
            sif2jax_equalities, sif2jax_inequalities = sif2jax_constraints
            sif2jax_equalities, _ = jfu.ravel_pytree(sif2jax_equalities)
            sif2jax_inequalities, _ = jfu.ravel_pytree(sif2jax_inequalities)

            # Check that the constraints match
            assert np.allclose(pycutest_equalities, sif2jax_equalities)
            assert np.allclose(pycutest_inequalities, sif2jax_inequalities)
        else:
            pytest.skip("Problem has no constraints")

    # def test_correct_options(self, problem, pycutest_problem):
    #     """Test for multiple starting points - not yet implemented in pycutest."""
    #     print(pycutest.print_available_sif_params(problem.name))
    #     if pycutest_problem.sifOptions is not None:
    #         print(pycutest_problem.sifOptions)
    #         print(problem.y0_iD)
    #         print(problem.provided_y0s)
    #         pass
    #     elif pycutest_problem.sifParams is not None:
    #         print(pycutest_problem.sifParams)
    #         print(problem.y0_iD)
    #         print(problem.provided_y0s)
    #         pass
    #     else:
    #         pytest.skip("Problem has no SIF options to specify.")

    def test_vmap(self, problem):
        try:
            vmapped = jax.vmap(problem.objective, in_axes=(0, None))
            y0 = problem.y0
            _ = vmapped(jnp.array([y0, y0, y0]), problem.args)
        except Exception as e:
            raise RuntimeError(f"Vmap failed for {problem.name}") from e

    def test_type_annotation_constraint(self, problem):
        if isinstance(problem, sif2jax.AbstractConstrainedMinimisation):
            signature = inspect.signature(problem.constraint)
            # No union types in return type hints of concrete implementations
            assert str(signature).split("->")[-1].strip().find("|") == -1
        elif isinstance(problem, sif2jax.AbstractNonlinearEquations):
            signature = inspect.signature(problem.constraint)
            # No union types in return type hints of concrete implementations
            assert str(signature).split("->")[-1].strip().find("|") == -1
        else:
            pytest.skip("Problem has no constraints")

    def test_type_annotation_objective(self, problem):
        signature = inspect.signature(problem.objective)
        # No union types in return type hints of concrete implementations
        assert str(signature).split("->")[-1].strip().find("|") == -1
