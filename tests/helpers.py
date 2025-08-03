import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import numpy as np
import pytest
import sif2jax


def try_except_evaluate(
    problem_name,
    sif2jax_func,
    pycutest_func,
    point,
    allclose_func=None,
    atol=1e-6,
    **kwargs,
):
    """Common function to evaluate both sif2jax and pycutest functions with errors.

    This function evaluates problem functions at a given point where we don't know if
    the function is actually defined at that point, or if evaluating it there causes a
    division by zero, an infinite value, or something else. It handles these cases
    gracefully and performs the comparison using the provided allclose function.

    These test cases serve to identify issues in the sif2jax problems that cannot be
    caught by evaluating the problems at a single point (e.g., only at the starting
    point).

    The function can be used to test objectives, gradients, Hessians, constraints, and
    constraint Jacobians. For example:
    - To test gradient: pass lambda x: jax.grad(problem.objective)(x, args) as
      sif2jax_func
    - To test Hessian: pass lambda x: jax.hessian(problem.objective)(x, args) as
      sif2jax_func

    **Arguments:**

    - `problem_name`: Name of the problem for error messages
    - `sif2jax_func`: Function to evaluate for sif2jax (e.g., objective, gradient,
        constraint)
    - `pycutest_func`: Corresponding function to evaluate for pycutest
    - `point`: Point at which to evaluate (e.g., zeros, ones, or starting point)
    - `allclose_func`: Optional function to compare values. Should have signature
        (pycutest_value, sif2jax_value, **kwargs). If None, uses np.allclose.
    - `atol`: Absolute tolerance for comparison (default: 1e-6). We use a more relaxed
        tolerance than np.allclose's default (1e-8) because:
        - JAX uses automatic differentiation for exact derivatives
        - pycutest may use hand-coded analytical derivatives or finite differences
        - Different numerical implementations can accumulate rounding errors differently
    - `**kwargs`: Additional keyword arguments passed to allclose_func

    Note:
        This function either passes (returns without error) or fails with pytest.fail.
        There is no return value. When both implementations fail with the same error
        type (ValueError or ZeroDivisionError), this is considered a success as both
        implementations agree on the behavior.
    """

    # NaN/Inf checker that handles any pytree structure
    def check_nan_inf(value):
        # Map nan/inf check across the tree, handling None values
        has_nan_or_inf = jax.tree.map(
            lambda x: jnp.isnan(x) | jnp.isinf(x) if x is not None else False, value
        )
        # Flatten and check if any are True
        flat, _ = jfu.ravel_pytree(has_nan_or_inf)
        return jnp.any(flat)

    # Evaluate pycutest
    pycutest_error = None
    pycutest_value = None
    pycutest_failed = False

    try:
        pycutest_value = pycutest_func(np.asarray(point))
        if check_nan_inf(pycutest_value):
            pycutest_error = ValueError("pycutest returned NaN or Inf.")
            pycutest_value = None
            pycutest_failed = True
    except (ZeroDivisionError, ValueError) as e:
        pycutest_error = e
        pycutest_failed = True

    # Evaluate sif2jax
    sif2jax_error = None
    sif2jax_value = None
    sif2jax_failed = False

    try:
        sif2jax_value = sif2jax_func(point)
        if check_nan_inf(sif2jax_value):
            sif2jax_error = ValueError("sif2jax returned NaN or Inf.")
            sif2jax_value = None
            sif2jax_failed = True
    except (ZeroDivisionError, ValueError) as e:
        sif2jax_error = e
        sif2jax_failed = True

    # Handle error cases
    if pycutest_failed and sif2jax_failed:
        if type(pycutest_error) != type(sif2jax_error):
            msg = (
                f"Errors differ for problem {problem_name} at point {point}: "
                f"pycutest_error={type(pycutest_error).__name__}, "
                f"sif2jax_error={type(sif2jax_error).__name__}"
            )
            pytest.fail(msg)
        else:
            # Both failed with same error - this is considered success
            return
    elif pycutest_failed or sif2jax_failed:
        if pycutest_failed and type(pycutest_error) is ValueError:
            # Special case: if pycutest fails but sif2jax returns a numerical value,
            # we assume that we have more robust numerics and that any discrepancy
            # can hopefully be found with another test. We cannot examine the
            # Fortran source code from here, so we would not have any information
            # relevant to fixing this case.
            return
        else:
            msg = (
                f"One implementation failed at point {point} for problem "
                f"{problem_name}: pycutest_failed={pycutest_failed}, "
                f"sif2jax_failed={sif2jax_failed}. "
            )
            if pycutest_failed:
                msg += f"pycutest_error={type(pycutest_error).__name__}"
            else:
                msg += f"sif2jax_error={type(sif2jax_error).__name__}"
            pytest.fail(msg)
    else:
        # Both succeeded - compare values
        assert pycutest_value is not None and sif2jax_value is not None

        if allclose_func is None:
            difference = pycutest_value - sif2jax_value
            abs_difference = jnp.abs(difference)
            msg = (
                f"Discrepancy for problem {problem_name} at point {point}. "
                f"The max. absolute difference at element "
                f"{jnp.argmax(abs_difference)} is {jnp.max(abs_difference)}."
            )
            assert np.allclose(pycutest_value, sif2jax_value, atol=atol), msg
        else:
            # Use custom comparison function (e.g., for constraints, Jacobians)
            # Pass atol along with other kwargs
            allclose_func(pycutest_value, sif2jax_value, atol=atol, **kwargs)


def constraints_allclose(
    pycutest_constraints, sif2jax_constraints, is_eq_cons, *, atol
):
    """Compare pycutest and sif2jax constraint values for equality.

    This function handles the different formats of constraints between pycutest and
    sif2jax:
    - pycutest returns a single array with all constraints
    - sif2jax returns a tuple of (equalities, inequalities)

    **Arguments:**

    - `pycutest_constraints`: Single array of all constraint values from pycutest
    - `sif2jax_constraints`: Tuple of (equalities, inequalities) from sif2jax
    - `is_eq_cons`: Boolean array indicating which constraints are equalities
    - `atol`: Absolute tolerance for comparison (required keyword argument)

    Raises:
        AssertionError: If constraints don't match within tolerance
    """
    import numpy as np

    # Parse pycutest constraints using boolean mask
    pycutest_equalities = pycutest_constraints[is_eq_cons]  # pyright: ignore
    pycutest_equalities = jnp.array(pycutest_equalities).squeeze()
    pycutest_inequalities = pycutest_constraints[~is_eq_cons]  # pyright: ignore
    pycutest_inequalities = jnp.array(pycutest_inequalities).squeeze()

    # Parse sif2jax constraints (already a tuple)
    sif2jax_equalities, sif2jax_inequalities = sif2jax_constraints

    if sif2jax_equalities is None:
        msg = "Discrepancy: sif2jax has no equality constraints but pycutest does."
        assert pycutest_equalities.size == 0, msg
    else:
        sif2jax_equalities = sif2jax_equalities.squeeze()
        difference = pycutest_equalities - sif2jax_equalities
        abs_difference = jnp.abs(difference)
        msg = (
            f"Discrepancy: sif2jax and pycutest equality constraints differ. The max. "
            f"absolute difference at element {jnp.argmax(abs_difference)} is "
            f"{jnp.max(abs_difference)}."
        )
        assert np.allclose(pycutest_equalities, sif2jax_equalities, atol=atol), msg

    if sif2jax_inequalities is None:
        msg = "Discrepancy: sif2jax has no inequality constraints but pycutest does."
        assert pycutest_inequalities.size == 0, msg
    else:
        sif2jax_inequalities = sif2jax_inequalities.squeeze()
        difference = pycutest_inequalities - sif2jax_inequalities
        abs_difference = jnp.abs(difference)
        msg = (
            f"Discrepancy: sif2jax and pycutest inequality constraints differ. The max."
            f" absolute difference at element {jnp.argmax(abs_difference)} is "
            f"{jnp.max(abs_difference)}."
        )
        assert np.allclose(pycutest_inequalities, sif2jax_inequalities, atol=atol), msg


def jacobians_allclose(pycutest_jac, sif2jax_jac, is_eq_cons, *, atol):
    """Compare pycutest and sif2jax constraint Jacobians for equality.

    This function handles the different formats of Jacobians between pycutest and
    sif2jax:
    - pycutest returns a single matrix with rows for all constraints
    - sif2jax returns a tuple of (equality_jac, inequality_jac)

    **Arguments:**

    - `pycutest_jac`: Single Jacobian matrix from pycutest
    - `sif2jax_jac`: Tuple of (equality_jac, inequality_jac) from sif2jax
    - `is_eq_cons`: Boolean array indicating which constraints are equalities
    - `atol`: Absolute tolerance for comparison (required keyword argument)

    Raises:
        AssertionError: If Jacobians don't match within tolerance
    """
    # Parse pycutest Jacobian - split into equalities and inequalities
    pycutest_eq_jac = pycutest_jac[is_eq_cons].squeeze()  # pyright: ignore
    pycutest_ineq_jac = pycutest_jac[~is_eq_cons].squeeze()  # pyright: ignore

    sif2jax_eq_jac, sif2jax_ineq_jac = sif2jax_jac

    if sif2jax_eq_jac is None:
        msg = "Discrepancy: sif2jax has no equality Jacobian but pycutest does."
        assert pycutest_eq_jac.size == 0, msg
    else:
        sif2jax_eq_jac = sif2jax_eq_jac.squeeze()
        difference = pycutest_eq_jac - sif2jax_eq_jac
        abs_difference = jnp.abs(difference)
        msg = (
            f"Discrepancy: sif2jax and pycutest equality Jacobians differ. The max. "
            f"absolute difference at element {jnp.argmax(abs_difference)} is "
            f"{jnp.max(abs_difference)}."
        )
        assert np.allclose(pycutest_eq_jac, sif2jax_eq_jac, atol=atol), msg
    if sif2jax_ineq_jac is None:
        msg = "Discrepancy: sif2jax has no inequality Jacobian but pycutest does."
        assert pycutest_ineq_jac.size == 0, msg
    else:
        sif2jax_ineq_jac = sif2jax_ineq_jac.squeeze()
        difference = pycutest_ineq_jac - sif2jax_ineq_jac
        abs_difference = jnp.abs(difference)
        msg = (
            f"Discrepancy: sif2jax and pycutest inequality Jacobians differ. The max. "
            f"absolute difference at element {jnp.argmax(abs_difference)} is "
            f"{jnp.max(abs_difference)}."
        )
        assert np.allclose(pycutest_ineq_jac, sif2jax_ineq_jac, atol=atol), msg


def has_constraints(problem):
    """Check if a problem has constraints.

    **Arguments:**

    - `problem`: A sif2jax problem instance

    **Returns:**

    - `bool`: True if the problem has constraints, False otherwise
    """
    return isinstance(
        problem,
        (sif2jax.AbstractConstrainedMinimisation, sif2jax.AbstractNonlinearEquations),
    )


def pycutest_jac_only(pycutest_problem):
    """Extract just the Jacobian from pycutest cons() function.

    Returns a function that takes a point and returns only the Jacobian matrix,
    discarding the constraint values.

    **Arguments:**

    - `pycutest_problem`: A pycutest problem instance

    **Returns:**

    - `function`: A function that takes a point and returns the Jacobian matrix
    """

    def jac_only(point):
        _, jac = pycutest_problem.cons(point, gradient=True)
        return jac

    return jac_only


def _sif2jax_hprod(problem, y):
    """AOT compiled Hessian-vector product for sif2jax problems.

    By compiling the evaluation, we avoid the materialisation of the Hessian matrix,
    which would result in OOM errors for large problems.

    **Arguments:**

    - `problem`: A sif2jax problem instance
    - `y`: The point at which to evaluate the Hessian

    **Returns:**

    - `Array`: The Hessian-vector product H(y) @ ones_like(y)
    """

    def hprod_(_y):
        return jax.hessian(problem.objective)(_y, problem.args) @ jnp.ones_like(_y)

    hprod = jax.jit(hprod_).lower(y).compile()
    return hprod(y)


def check_hprod_allclose(problem, pycutest_problem, point, *, atol=1e-6):
    """Compute and compare pycutest and sif2jax Hessian-vector products for equality.

    This function computes Hessian-vector products H(point) @ ones for both pycutest and
    sif2jax implementations and compares them, where H(point) is the Hessian evaluated
    at the given point and ones is a vector of all ones.

    **Arguments:**

    - `problem`: A sif2jax problem instance
    - `pycutest_problem`: A pycutest problem instance
    - `point`: The point at which to evaluate the Hessian
    - `atol`: Absolute tolerance for comparison (default: 1e-6)

    **Raises:**

    - `AssertionError`: If Hessian-vector products don't match within tolerance

    **Note:**

    pycutest implements its hprod method for the Hessian only if the problem is
    unconstrained. For constrained problems, the Hessian used by pycutest is always the
    Hessian of the Lagrangian, which requires multiplier values.
    """
    # Compute both hprods
    # Note: pycutest.hprod(p, x) computes H(x) @ p
    # We use a vector of ones and evaluate the Hessian at the given point
    pycutest_hprod = pycutest_problem.hprod(np.ones_like(point), np.asarray(point))
    sif2jax_hprod = _sif2jax_hprod(problem, point)

    # Check for NaN or inf values
    pycutest_has_nonfinite = ~jnp.isfinite(pycutest_hprod)
    sif2jax_has_nonfinite = ~jnp.isfinite(sif2jax_hprod)

    pycutest_any_nonfinite = jnp.any(pycutest_has_nonfinite)
    sif2jax_any_nonfinite = jnp.any(sif2jax_has_nonfinite)

    if pycutest_any_nonfinite or sif2jax_any_nonfinite:
        if pycutest_any_nonfinite and sif2jax_any_nonfinite:
            # Both have NaN/inf - check if they're in the same places
            nonfinite_diff = jnp.sum(
                pycutest_has_nonfinite.astype(int) - sif2jax_has_nonfinite.astype(int)
            )
            if nonfinite_diff == 0:
                # NaN/inf in same places - test passes
                return
            else:
                # NaN/inf in different places - test fails
                pycutest_first_idx = jnp.argmax(pycutest_has_nonfinite)
                sif2jax_first_idx = jnp.argmax(sif2jax_has_nonfinite)
                msg = (
                    f"Hessian-vector products contain NaN or inf values for different "
                    f"elements in problem {problem.name}. "
                    f"First non-finite index in pycutest: {pycutest_first_idx}, "
                    f"first non-finite index in sif2jax: {sif2jax_first_idx}."
                )
                pytest.fail(msg)
        else:
            # Only one has NaN/inf - provide informative message
            if pycutest_any_nonfinite:
                first_idx = jnp.argmax(pycutest_has_nonfinite)
                msg = (
                    f"Only pycutest Hessian-vector product contains NaN or inf values "
                    f"for problem {problem.name}. First non-finite index: {first_idx}."
                )
            else:
                first_idx = jnp.argmax(sif2jax_has_nonfinite)
                msg = (
                    f"Only sif2jax Hessian-vector product contains NaN or inf values "
                    f"for problem {problem.name}. First non-finite index: {first_idx}."
                )
            pytest.fail(msg)
    else:
        # Neither has NaN/inf - proceed with normal comparison
        difference = pycutest_hprod - sif2jax_hprod
        abs_difference = jnp.abs(difference)
        msg = (
            f"Mismatch in Hessian-vector product for {problem.name}. "
            f"The max. absolute difference is at element {jnp.argmax(abs_difference)} "
            f"with a value of {jnp.max(abs_difference)}."
        )
        assert np.allclose(pycutest_hprod, sif2jax_hprod, atol=atol), msg
