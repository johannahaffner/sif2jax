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

if __name__ == "__main__":

    name = "AKIVA"
    p_ref = pycutest.import_problem(name)
    p = sif2jax.problems[0]

    # unconstrained mapping to cyipopt simple
    obj_ref      = p_ref.obj
    obj_grad_ref = p_ref.grad                 # gradient of objective
    obj_hess_ref = lambda x: p_ref.ihess(x)                 # dense Hessian, (n,n)
    y0_ref = p_ref.x0

    args = p.args()
    y0 = p.y0()
    obj = p.objective(y0, args)
    obj_grad = jax.grad(p.objective)(y0, args)          # gradient of objective
    obj_hess = jax.hessian(p.objective)(y0, args)       # dense Hessian, (n,n)

    print(obj)
    print(obj_grad)
    print(obj_hess)

    # cyipopt problem
    f = lambda x: p.objective(x, args)

    # jit the functions
    obj_jit = f

    # build the derivatives and jit them
    obj_grad = jax.grad(obj_jit)  # objective gradient
    obj_hess = jax.jacrev(jax.jacfwd(obj_jit)) # objective hessian

    # print("starting compile...")
    # obj_jit(p.y0())
    # obj_grad(p.y0())
    # obj_hess(p.y0())

    maxtol_coeff = 10
    maxiter = 100

    # IPOPT minimize call directly with jax functions when doing dense.
    for i in range(maxtol_coeff):
        tol = 10**(-i)
        print(f"Running cyipopt with tol = {tol}")
        result = cyipopt.minimize_ipopt(
            fun=obj_jit, 
            x0=p.y0(), 
            jac=obj_grad, 
            hess=obj_hess, 
            tol=tol, 
            options={'disp': 5, 'maxiter': 100}
        )
        if result.success is False and i > 0:
            print(f"Highest tol with successful optimization in {maxiter} iterations: {10**(-i+1)}")
            break
        if result.success is False and i == 0:
            print(f"Failed to converge with tol = {tol}")
            break

        # print("Result from cyipopt:")
        # print(result)
        # print("Objective value:", obj_jit(result.x))
        # print("Gradient value:", obj_grad(result.x))
        # print("Hessian value:", obj_hess(result.x))
        # print("Final point:", result.x)
    else:
        print(f"Converged with maximum tol: {tol}")

    # repeat with pycutest
    for i in range(maxtol_coeff):
        tol = 10**(-i)
        print(f"Running pycutest with tol = {tol}")
        result_ref = cyipopt.minimize_ipopt(
            fun=obj_ref, 
            x0=y0_ref, 
            jac=obj_grad_ref, 
            hess=obj_hess_ref, 
            tol=tol, 
            options={'disp': 5, 'maxiter': maxiter}
        )
        if result_ref.success is False and i > 0:
            print(f"Highest tol with successful optimization in {maxiter} iterations: {10**(-i+1)}")
            break
        if result_ref.success is False and i == 0:
            print(f"Failed to converge with tol = {tol}")
            break

        # print("Result from pycutest:")
        # print(result_ref)
        # print("Objective value:", obj_ref(result_ref.x))
        # print("Gradient value:", obj_grad_ref(result_ref.x))
        # print("Hessian value:", obj_hess_ref(result_ref.x))
        # print("Final point:", result_ref.x)
    else:
        print(f"Converged with maximum tol: {tol}")

    print("fin")
