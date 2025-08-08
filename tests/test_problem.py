import inspect

import equinox as eqx
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import numpy as np
import pycutest  # pyright: ignore[reportMissingImports]  - test runs in container
import pytest  # pyright: ignore[reportMissingImports]  - test runs in container
import sif2jax

from .helpers import (
    check_hprod_allclose,
    constraints_allclose,
    has_constraints,
    jacobians_allclose,
    pycutest_jac_only,
    try_except_evaluate,
)


@pytest.fixture(scope="class")
def clear_caches():
    # Setup
    yield
    eqx.clear_caches()
    # Teardown


@pytest.mark.usefixtures("clear_caches")
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
        return pycutest.import_problem(problem.name, drop_fixed_variables=False)

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
        try_except_evaluate(
            problem.__class__.__name__,  # sif2jax name (e.g. TENFOLD not 10FOLD)
            lambda x: problem.objective(x, problem.args),
            pycutest_problem.obj,
            jnp.zeros_like(problem.y0),
        )

    def test_correct_objective_ones_vector(self, problem, pycutest_problem):
        try_except_evaluate(
            problem.__class__.__name__,  # sif2jax name (e.g. TENFOLD not 10FOLD)
            lambda x: problem.objective(x, problem.args),
            pycutest_problem.obj,
            jnp.ones_like(problem.y0),
        )

    def test_correct_gradient_at_start(self, problem, pycutest_problem):
        pycutest_gradient = pycutest_problem.grad(pycutest_problem.x0)
        sif2jax_gradient = jax.grad(problem.objective)(problem.y0, problem.args)
        assert np.allclose(pycutest_gradient, sif2jax_gradient)

    def test_correct_gradient_zero_vector(self, problem, pycutest_problem):
        try_except_evaluate(
            problem.__class__.__name__,  # sif2jax name (e.g. TENFOLD not 10FOLD)
            lambda x: jax.grad(problem.objective)(x, problem.args),
            pycutest_problem.grad,
            jnp.zeros_like(problem.y0),
        )

    def test_correct_gradient_ones_vector(self, problem, pycutest_problem):
        try_except_evaluate(
            problem.__class__.__name__,  # sif2jax name (e.g. TENFOLD not 10FOLD)
            lambda x: jax.grad(problem.objective)(x, problem.args),
            pycutest_problem.grad,
            jnp.ones_like(problem.y0),
        )

    def test_correct_hessian_at_start(self, problem, pycutest_problem):
        if problem.num_variables() <= 1000:
            pycutest_hessian = pycutest_problem.ihess(np.asarray(problem.y0))
            sif2jax_hessian = jax.hessian(problem.objective)(problem.y0, problem.args)
            assert np.allclose(pycutest_hessian, sif2jax_hessian)
        else:
            if isinstance(problem, sif2jax.AbstractUnconstrainedMinimisation):
                # Note that if the starting point y0 is all zeros (which is the case for
                # some problems), then the Hessian-vector product defined as Hess @ y0
                # is trivial and uninformative.
                if problem.num_variables() <= 10_000:
                    check_hprod_allclose(problem, pycutest_problem, problem.y0)
                else:
                    pytest.skip(
                        "Hessian-vector product test skipped for very large "
                        "problems (n >= 10,000) due to memory constraints."
                    )
            else:
                # pycutest implements its hprod method for the Hessian only if the
                # problem is unconstrained. Otherwise the Hessian used in this method
                # is the Hessian of the Lagrangian, and multiplier values must be
                # provided. We only test the Hessians in this method here.
                msg = (
                    "Hessian-vector product test not implemented for constrained "
                    "problems that require values for the dual variables to compute "
                    "the Hessian of the Lagrangian. "
                )
                pytest.skip(msg)

    def test_correct_hessian_zero_vector(self, problem, pycutest_problem):
        if problem.num_variables() <= 1000:
            try_except_evaluate(
                problem.__class__.__name__,  # sif2jax name (e.g. TENFOLD not 10FOLD)
                lambda x: jax.hessian(problem.objective)(x, problem.args),
                pycutest_problem.ihess,
                jnp.zeros_like(problem.y0),
            )
        else:
            if isinstance(problem, sif2jax.AbstractUnconstrainedMinimisation):
                if problem.num_variables() <= 10_000:
                    check_hprod_allclose(
                        problem, pycutest_problem, jnp.zeros_like(problem.y0)
                    )
                else:
                    pytest.skip(
                        "Hessian-vector product test skipped for very large "
                        "problems (n >= 10,000) due to memory constraints."
                    )
            else:
                # pycutest implements its hprod method for the Hessian only if the
                # problem is unconstrained. Otherwise the Hessian used in this method
                # is the Hessian of the Lagrangian, and multiplier values must be
                # provided. We only test the Hessians in this method here.
                msg = (
                    "Hessian-vector product test not implemented for constrained "
                    "problems that require values for the dual variables to compute "
                    "the Hessian of the Lagrangian. "
                )
                pytest.skip(msg)

    def test_correct_hessian_ones_vector(self, problem, pycutest_problem):
        if problem.num_variables() <= 1000:
            try_except_evaluate(
                problem.__class__.__name__,  # sif2jax name (e.g. TENFOLD not 10FOLD)
                lambda x: jax.hessian(problem.objective)(x, problem.args),
                pycutest_problem.ihess,
                jnp.ones_like(problem.y0),
            )
        else:
            if isinstance(problem, sif2jax.AbstractUnconstrainedMinimisation):
                if problem.num_variables() <= 10_000:
                    check_hprod_allclose(
                        problem, pycutest_problem, jnp.ones_like(problem.y0)
                    )
                else:
                    pytest.skip(
                        "Hessian-vector product test skipped for very large "
                        "problems (n >= 10,000) due to memory constraints."
                    )
            else:
                # pycutest implements its hprod method for the Hessian only if the
                # problem is unconstrained. Otherwise the Hessian used in this method
                # is the Hessian of the Lagrangian, and multiplier values must be
                # provided. We only test the Hessians in this method here.
                msg = (
                    "Hessian-vector product test not implemented for constrained "
                    "problems that require values for the dual variables to compute "
                    "the Hessian of the Lagrangian. "
                )
                pytest.skip(msg)

    def test_correct_constraint_dimensions(self, problem, pycutest_problem):
        # TODO: Should we even return something like this for unconstrained problems?
        # Think about this: perhaps it is better if this method would not exist for the
        # unconstrained problems at all.
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

    def test_nontrivial_constraints(self, problem):
        if has_constraints(problem):
            equalities, inequalities = problem.constraint(problem.y0)
            # Check that the problem is not mistakenly classified as constrained
            # If both elements of the tuple are None, the problem is unconstrained or
            # bound constrained, and should inherit from a different parent class.
            assert equalities is not None or inequalities is not None

            # We never return empty arrays, if there are no inequality or equality
            # constraints then None is returned.
            if equalities is not None:
                msg = "Equality constraints should be None if there are no equalities."
                assert equalities.size > 0, msg
            if inequalities is not None:
                msg = (
                    "Inequality constraints should be None if there are no such "
                    "constraints."
                )
                assert inequalities.size > 0, msg

    def test_nontrivial_bounds(self, problem):
        if (
            isinstance(problem, sif2jax.AbstractConstrainedMinimisation)
            # AbstractConstrainedMinimisation includes subclass for quadratic problems
            or isinstance(problem, sif2jax.AbstractBoundedMinimisation)
            or isinstance(problem, sif2jax.AbstractNonlinearEquations)
        ):
            bounds = problem.bounds
            if bounds is not None:
                lower, upper = bounds
                assert lower is not None and upper is not None
                # If bounds are not None, then at least one element of `y` should have
                # a nontrivial (finite) bound.
                # Otherwise the bounds method should return None.
                has_finite_lower = jnp.any(jnp.isfinite(lower))
                has_finite_upper = jnp.any(jnp.isfinite(upper))
                assert has_finite_lower or has_finite_upper

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
        if has_constraints(problem):
            assert pycutest_problem.m > 0, "Problem should have constraints."

            pycutest_constraints = pycutest_problem.cons(pycutest_problem.x0)
            sif2jax_constraints = problem.constraint(problem.y0)

            # Check that the constraints match
            constraints_allclose(
                pycutest_constraints,
                sif2jax_constraints,
                pycutest_problem.is_eq_cons,
                atol=1e-6,
            )
        else:
            pytest.skip("Problem has no constraints")

    def test_correct_constraints_zero_vector(self, problem, pycutest_problem):
        if has_constraints(problem):
            try_except_evaluate(
                problem.__class__.__name__,
                lambda x: problem.constraint(x),
                pycutest_problem.cons,
                jnp.zeros_like(problem.y0),
                allclose_func=lambda p, s, **kwargs: constraints_allclose(
                    p, s, pycutest_problem.is_eq_cons, **kwargs
                ),
            )
        else:
            pytest.skip("Problem has no constraints")

    def test_correct_constraints_ones_vector(self, problem, pycutest_problem):
        if has_constraints(problem):
            try_except_evaluate(
                problem.__class__.__name__,
                lambda x: problem.constraint(x),
                pycutest_problem.cons,
                jnp.ones_like(problem.y0),
                allclose_func=lambda p, s, **kwargs: constraints_allclose(
                    p, s, pycutest_problem.is_eq_cons, **kwargs
                ),
            )
        else:
            pytest.skip("Problem has no constraints")

    def test_correct_constraint_jacobian_at_start(self, problem, pycutest_problem):
        if has_constraints(problem):
            constraints, _ = jfu.ravel_pytree(problem.constraint(problem.y0))
            if problem.y0.size * constraints.size < 1_000_000:
                try_except_evaluate(
                    problem.__class__.__name__,
                    lambda p: jax.jacfwd(lambda x: problem.constraint(x))(p),
                    pycutest_jac_only(pycutest_problem),
                    problem.y0,
                    allclose_func=lambda p, s, **kwargs: jacobians_allclose(
                        p, s, pycutest_problem.is_eq_cons, **kwargs
                    ),
                )
            else:
                pytest.skip("Skip (dense) Jacobian test for large problems.")
        else:
            pytest.skip("Problem has no constraints")

    def test_correct_constraint_jacobian_zero_vector(self, problem, pycutest_problem):
        if has_constraints(problem):
            constraints, _ = jfu.ravel_pytree(problem.constraint(problem.y0))
            if problem.y0.size * constraints.size < 1_000_000:
                try_except_evaluate(
                    problem.__class__.__name__,
                    lambda p: jax.jacfwd(lambda x: problem.constraint(x))(p),
                    pycutest_jac_only(pycutest_problem),
                    jnp.zeros_like(problem.y0),
                    allclose_func=lambda p, s, **kwargs: jacobians_allclose(
                        p, s, pycutest_problem.is_eq_cons, **kwargs
                    ),
                )
            else:
                pytest.skip("Skip (dense) Jacobian test for large problems.")
        else:
            pytest.skip("Problem has no constraints")

    def test_correct_constraint_jacobian_ones_vector(self, problem, pycutest_problem):
        if has_constraints(problem):
            constraints, _ = jfu.ravel_pytree(problem.constraint(problem.y0))
            if problem.y0.size * constraints.size < 1_000_000:
                try_except_evaluate(
                    problem.__class__.__name__,
                    lambda p: jax.jacfwd(lambda x: problem.constraint(x))(p),
                    pycutest_jac_only(pycutest_problem),
                    jnp.ones_like(problem.y0),
                    allclose_func=lambda p, s, **kwargs: jacobians_allclose(
                        p, s, pycutest_problem.is_eq_cons, **kwargs
                    ),
                )
            else:
                pytest.skip("Skip (dense) Jacobian test for large problems.")
        else:
            pytest.skip("Problem has no constraints")

    def test_with_sparse_hessian(self, problem, pycutest_problem):
        """This test checks if the nonzero elements of the Hessian matrix match the ones
        given by pycutest in BCOO representation.
        """
        # print(pycutest_problem.ihess(np.asarray(problem.y0)))
        # print(pycutest_problem.isphess(np.asarray(problem.y0)))
        pass

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
        if has_constraints(problem):
            signature = inspect.signature(problem.constraint)
            # No union types in return type hints of concrete implementations
            assert str(signature).split("->")[-1].strip().find("|") == -1
        else:
            pytest.skip("Problem has no constraints")

    def test_type_annotation_objective(self, problem):
        signature = inspect.signature(problem.objective)
        # No union types in return type hints of concrete implementations
        assert str(signature).split("->")[-1].strip().find("|") == -1
