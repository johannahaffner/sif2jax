"""
This script aims to run both the pycutest problem and the sif2jax problem through
cyipopts dense ipopt interface to conduct an end-to-end test. The resulting output,
however suboptimal should be the same for both problems.
"""

import pycutest
import numpy as np
import cyipopt
import sif2jax
import jax
jax.config.update("jax_enable_x64", True)  # ESSENTIAL
import jax.numpy as jnp
import pytest
import equinox as eqx

def tree_numeric_allclose(tree1, tree2, *, rtol=1e-5, atol=1e-8):
    """
    Compare two pytrees of the *same structure*.

    • Numeric leaves (ndarrays, JAX arrays, NumPy scalars, Python floats / ints)
      are compared with jnp.allclose.
    • All other leaf types are ignored (treated as equal).

    Returns
    -------
    bool
        True  → every comparable numeric leaf is close enough.
        False → at least one numeric leaf differs beyond tolerance.

    Raises
    ------
    ValueError
        If the two pytrees do not share the same treedef.
    """

    flat1, treedef1 = jax.tree_util.tree_flatten(tree1)
    flat2, treedef2 = jax.tree_util.tree_flatten(tree2)

    if treedef1 != treedef2:
        raise ValueError("Pytrees have different structures and cannot be compared")

    def _is_numeric(x):
        return (
            isinstance(x, (jnp.ndarray, np.ndarray))
            or np.isscalar(x)
            or isinstance(x, (int, float, complex))
        )

    def _leaf_close(a, b):
        if _is_numeric(a) and _is_numeric(b):
            return bool(jnp.allclose(a, b, rtol=rtol, atol=atol))
        # non‑numeric → ignore
        return True

    return all(_leaf_close(a, b) for a, b in zip(flat1, flat2))

@pytest.mark.parametrize("problem", sif2jax.problems)
def e2e_dense_ipopt_test(problem):
    """
    Run the pycutest problem and the sif2jax problem through cyipopts dense ipopt interface
    to conduct an end-to-end test. The resulting output, however suboptimal should be the same
    for both problems.
    """

    # Import the pycutest problem
    p_ref = pycutest.import_problem(problem.name())
    p = problem

    # Define the objective function, gradient, and hessian for both problems
    obj_ref      = p_ref.obj
    obj_grad_ref = p_ref.grad
    obj_hess_ref = p_ref.ihess

    obj = lambda x: p.objective(x, p.args())
    obj_grad = jax.jacobian(obj)
    obj_hess = jax.hessian(obj)

    # Set tolerance and maximum iterations
    maxtol_coeff = 10
    maxiter = 100

    for i in range(maxtol_coeff):
        tol = 10**(-i)
        print(f"Running cyipopt with tol = {tol}")
        result = cyipopt.minimize_ipopt(
            fun=obj, 
            x0=p.y0(), 
            jac=obj_grad, 
            hess=obj_hess, 
            tol=tol, 
            options={'disp': 5, 'maxiter': maxiter}
        )
        result_ref = cyipopt.minimize_ipopt(
            fun=obj_ref, 
            x0=p_ref.x0, 
            jac=obj_grad_ref, 
            hess=obj_hess_ref, 
            tol=tol, 
            options={'disp': 5, 'maxiter': maxiter}
        )

        # assert that all numerical values in both scipy results objects are equal
        assert tree_numeric_allclose(result, result_ref)

        if result.success is False and i > 0:
            print(f"Highest tol with successful optimization in {maxiter} iterations: {10**(-i+1)}")
            break
        if result.success is False and i == 0:
            print(f"Failed to converge with tol = {tol}")
            break

    else:
        print(f"Converged with maximum tol: {tol}")


if __name__ == "__main__":

    for problem in sif2jax.problems:
        print(f"Running end-to-end test for problem: {problem.name()}")
        e2e_dense_ipopt_test(problem)

    name = "AKIVA"
    p_ref = pycutest.import_problem(name)
    p = sif2jax.problems[0]

    obj_ref      = p_ref.obj
    obj_grad_ref = p_ref.grad
    obj_hess_ref = p_ref.ihess

    obj = lambda x: p.objective(x, p.args())
    obj_grad = jax.jacobian(obj)
    obj_hess = jax.hessian(obj)

    maxtol_coeff = 10
    maxiter = 100

    for i in range(maxtol_coeff):
        tol = 10**(-i)
        print(f"Running cyipopt with tol = {tol}")
        result = cyipopt.minimize_ipopt(
            fun=obj, 
            x0=p.y0(), 
            jac=obj_grad, 
            hess=obj_hess, 
            tol=tol, 
            options={'disp': 5, 'maxiter': maxiter}
        )
        result_ref = cyipopt.minimize_ipopt(
            fun=obj_ref, 
            x0=p_ref.x0, 
            jac=obj_grad_ref, 
            hess=obj_hess_ref, 
            tol=tol, 
            options={'disp': 5, 'maxiter': maxiter}
        )

        # assert that all numerical values in both scipy results objects are equal
        assert tree_numeric_allclose(result, result_ref)

        if result.success is False and i > 0:
            print(f"Highest tol with successful optimization in {maxiter} iterations: {10**(-i+1)}")
            break
        if result.success is False and i == 0:
            print(f"Failed to converge with tol = {tol}")
            break

    else:
        print(f"Converged with maximum tol: {tol}")

    print("fin")
