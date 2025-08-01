import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import numpy as np
import pytest


def _try_except_evaluate(
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

        # Use provided allclose function or default to np.allclose
        if allclose_func is None:
            # Default comparison for simple values
            # Pass atol to np.allclose, along with any other kwargs
            assert np.allclose(pycutest_value, sif2jax_value, atol=atol, **kwargs)
        else:
            # Use custom comparison function (e.g., for constraints, Jacobians)
            # Pass atol along with other kwargs
            allclose_func(pycutest_value, sif2jax_value, atol=atol, **kwargs)


def _constraints_allclose(
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
    sif2jax_equalities, _ = jfu.ravel_pytree(sif2jax_equalities)
    sif2jax_inequalities, _ = jfu.ravel_pytree(sif2jax_inequalities)

    # Check that the constraints match
    assert np.allclose(pycutest_equalities, sif2jax_equalities, atol=atol)
    assert np.allclose(pycutest_inequalities, sif2jax_inequalities, atol=atol)


def _jacobians_allclose(pycutest_jac, sif2jax_jac, is_eq_cons, *, atol):
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
        msg = (
            f"Discrepancy: sif2jax and pycutest equality Jacobians differ. The max. "
            f"difference at element {jnp.argmax(difference)} is {jnp.max(difference)}."
        )
        assert np.allclose(pycutest_eq_jac, sif2jax_eq_jac, atol=atol), msg
    if sif2jax_ineq_jac is None:
        msg = "Discrepancy: sif2jax has no inequality Jacobian but pycutest does."
        assert pycutest_ineq_jac.size == 0, msg
    else:
        sif2jax_ineq_jac = sif2jax_ineq_jac.squeeze()
        difference = pycutest_ineq_jac - sif2jax_ineq_jac
        msg = (
            f"Discrepancy: sif2jax and pycutest inequality Jacobians differ. The max. "
            f"difference at element {jnp.argmax(difference)} is {jnp.max(difference)}."
        )
        assert np.allclose(pycutest_ineq_jac, sif2jax_ineq_jac, atol=atol), msg
