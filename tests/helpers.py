import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import numpy as np
import pytest
import sif2jax
from jax.experimental.sparse import BCOO


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

def _sif2jax_sparse_hprod(problem, y, hess_coo):
    obj = lambda x: problem.objective(x, problem.args)
    sphess_fun = sparse_hessian(obj, hess_coo, [1,y.size,y.size]) # [fun out size, inp size, inp size] always for vector input vector output functions
    hprod = sphess_fun(y) @ jnp.ones_like(y)
    return hprod

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
    pycutest_nonfinite = ~jnp.isfinite(pycutest_hprod)
    sif2jax_nonfinite = ~jnp.isfinite(sif2jax_hprod)

    if jnp.any(pycutest_nonfinite) and jnp.any(sif2jax_nonfinite):
        # Both have NaN/inf - check if they're in the same places
        nonfinite_diff = jnp.sum(
            pycutest_nonfinite.astype(int) - sif2jax_nonfinite.astype(int)
        )
        if nonfinite_diff == 0:
            # NaN/inf in same places - test passes
            return
        else:
            # NaN/inf in different places - test fails
            pycutest_first_idx = jnp.argmax(pycutest_nonfinite)
            sif2jax_first_idx = jnp.argmax(sif2jax_nonfinite)
            msg = (
                f"Hessian-vector products contain NaN or inf values for different "
                f"elements in problem {problem.name}. "
                f"First non-finite index in pycutest: {pycutest_first_idx}, "
                f"first non-finite index in sif2jax: {sif2jax_first_idx}."
            )
            pytest.fail(msg)
    elif jnp.any(pycutest_nonfinite):
        # Only pycutest has NaN/inf
        first_idx = jnp.argmax(pycutest_nonfinite)
        msg = (
            f"Only pycutest Hessian-vector product contains NaN or inf values "
            f"for problem {problem.name}. First non-finite index: {first_idx}."
        )
        pytest.fail(msg)
    elif jnp.any(sif2jax_nonfinite):
        # Only sif2jax has NaN/inf
        first_idx = jnp.argmax(sif2jax_nonfinite)
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

def sparse_jacobian(f, jac_coo, out_shape):

    if len(jac_coo) == 0:
        # indicates that there isn't a function or no known Jacobian pattern
        return lambda x: BCOO([jnp.array([]), jnp.zeros([0, len(out_shape)], dtype=jnp.int32)], shape=out_shape)

    # Ensure jac_coo is a JAX array of int
    jac_coo = jnp.array(jac_coo, dtype=jnp.int32)
    ncols = jac_coo.shape[1]
    # adjust jac_coo to allow for jittability in partial when slicing y on return statement
    # nrows = jac_coo.shape[0]
    # jac_coo = jnp.hstack([jnp.zeros([nrows,1], dtype=jnp.int32), jac_coo]) if ncols == 1 else jac_coo
    
    # sometimes function is scalar output, giving a leading 1 in the out_shape - we remove it
    if out_shape[0] == 1:
        out_shape = out_shape[1:]

    # CASE 1: vector-valued function: R^n -> R^m
    #         Each row in jac_coo is [out_idx, x_idx]
    if ncols > 1:  
        def partial_out_j(x, out_idx, x_idx):
            """
            Computes d(f_out_idx)(x)/dx_x_idx = partial derivative of the out_idx-th
            component of f w.r.t. the x_idx-th component of x.
            """
            # grad_f_out is the gradient of f(...)[out_idx], i.e. R^n -> scalar
            grad_f_out = jax.grad(lambda z: f(z)[out_idx])(x)
            return grad_f_out[x_idx]

        def jac_fn(x):
            """
            Evaluate all requested Jacobian entries at x,
            and store them in a BCOO with coords=jac_coo, data=the partials.
            """
            def single_entry(coord):
                out_idx, x_idx = coord
                return partial_out_j(x, out_idx, x_idx)

            # Vectorize over rows of jac_coo to get all partial derivatives
            out = jax.vmap(single_entry)(jac_coo)
            # In the style of your sparse_hessian, we'll store shape=jac_coo.shape
            return BCOO((out, jac_coo), shape=out_shape)

    # CASE 2: scalar-valued function: R^n -> R
    #         Each row in jac_coo is [x_idx]
    elif ncols == 1:
        def partial_j(x, x_idx):
            """
            Computes d f(x)/dx_x_idx for a scalar function f.
            """
            grad_f = jax.grad(f)(x)  # shape (n,)
            return grad_f[x_idx]

        def jac_fn(x):
            def single_entry(coord):
                x_idx = coord[0]
                return partial_j(x, x_idx)

            out = jax.vmap(single_entry)(jac_coo)
            return BCOO((out, jac_coo), shape=out_shape)

    else:
        # Not a recognized pattern
        raise ValueError(
            f"jac_coo should have either 1 column (scalar f) or 2 columns (vector f). "
            f"Got shape={out_shape}."
        )

    return jac_fn

def sparse_hessian(f, hes_coo, out_shape):

    if len(hes_coo) == 0:
        # indicates that there isn't a function / or no known Hessian pattern
        return lambda x: BCOO([jnp.array([]), jnp.zeros([0, len(out_shape)], dtype=jnp.int32)], shape=out_shape)

    # Ensure hes_coo is a JAX array of int
    hes_coo = jnp.array(hes_coo, dtype=jnp.int32)
    ncols = hes_coo.shape[1]

    # sometimes function is scalar output, giving a leading 1 in the out_shape - we remove it
    if out_shape[0] == 1:
        out_shape = out_shape[1:]

    if ncols > 2:

        def partial_out_ij(x, out_idx, i, j):
            """
            Compute d^2 f_{out_idx}(x) / (dx_i dx_j).
            That is: first derivative w.r.t. x_i, then derivative of that result w.r.t. x_j.
            """

            # 1) Define a function g_i(u) = df_{out_idx}(u)/dx_i
            #    i.e., pick out the i-th component from jax.grad(f_{out_idx})(u).
            #    f_{out_idx}(u) means f(u)[out_idx].
            def g_i(u):
                return jax.grad(lambda z: f(z)[out_idx])(u)[i]

            # 2) Now take derivative of g_i w.r.t. x_j
            #    i.e. second partial derivative.
            return jax.grad(g_i)(x)[j]

        def hess_fn(x):
            """
            Evaluate all requested Hessian entries at x and return them as a 1D array.
            """
            def single_entry(coord):
                out_idx, i, j = coord
                return partial_out_ij(x, out_idx, i, j)

            # Vectorize over rows of hes_coo
            out = jax.vmap(single_entry)(hes_coo)
            return BCOO([out, hes_coo], shape=out_shape)
        
    elif ncols == 2:

        def partial_ij(x, i, j):
            """
            Compute d^2 f(x) / (dx_i dx_j).
            That is: first derivative w.r.t. x_i, then derivative of that result w.r.t. x_j.
            """
            # 1) g_i(u) = derivative of f(u) w.r.t. x_i
            #    jax.grad(f)(u) is a vector, pick out component i
            def g_i(u):
                return jax.grad(f)(u)[i]

            # 2) derivative of g_i(u) w.r.t. x_j
            return jax.grad(g_i)(x)[j]

        def hess_fn(x):
            """
            Evaluate all requested Hessian entries at x and return them as a 1D array.
            """
            def single_entry(coord):
                i, j = coord
                return partial_ij(x, i, j)

            # Vectorize over rows of hes_coo
            out = jax.vmap(single_entry)(hes_coo)
            return BCOO([out, hes_coo], shape=out_shape)

    return hess_fn

def check_sparse_hprod_allclose(problem, pycutest_problem, point, *, atol=1e-6):

    pycutest_hprod = pycutest_problem.hprod(np.ones_like(point), np.asarray(point))
    sphess = pycutest_problem.isphess(np.asarray(point))
    hess_coo = jnp.vstack(sphess.coords).T
    sif2jax_hprod = _sif2jax_sparse_hprod(problem, point, hess_coo)

    # Check for NaN or inf values
    pycutest_nonfinite = ~jnp.isfinite(pycutest_hprod)
    sif2jax_nonfinite = ~jnp.isfinite(sif2jax_hprod)

    if jnp.any(pycutest_nonfinite) and jnp.any(sif2jax_nonfinite):
        # Both have NaN/inf - check if they're in the same places
        nonfinite_diff = jnp.sum(
            pycutest_nonfinite.astype(int) - sif2jax_nonfinite.astype(int)
        )
        if nonfinite_diff == 0:
            # NaN/inf in same places - test passes
            return
        else:
            # NaN/inf in different places - test fails
            pycutest_first_idx = jnp.argmax(pycutest_nonfinite)
            sif2jax_first_idx = jnp.argmax(sif2jax_nonfinite)
            msg = (
                f"Hessian-vector products contain NaN or inf values for different "
                f"elements in problem {problem.name}. "
                f"First non-finite index in pycutest: {pycutest_first_idx}, "
                f"first non-finite index in sif2jax: {sif2jax_first_idx}."
            )
            pytest.fail(msg)
    elif jnp.any(pycutest_nonfinite):
        # Only pycutest has NaN/inf
        first_idx = jnp.argmax(pycutest_nonfinite)
        msg = (
            f"Only pycutest Hessian-vector product contains NaN or inf values "
            f"for problem {problem.name}. First non-finite index: {first_idx}."
        )
        pytest.fail(msg)
    elif jnp.any(sif2jax_nonfinite):
        # Only sif2jax has NaN/inf
        first_idx = jnp.argmax(sif2jax_nonfinite)
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
