"""
This script aims to run both the pycutest problem and the sif2jax problem through
cyipopts dense ipopt interface to conduct an end-to-end test. The resulting output,
however suboptimal should be the same for both problems.
"""
# pyright: reportMissingImports=false, reportUnboundVariable=false

import cyipopt  # pyright: ignore[reportMissingImports]
import jax
import numpy as np
import pycutest
import sif2jax


jax.config.update("jax_enable_x64", True)  # ESSENTIAL
import io  # for capturing stdout

import pytest
from wurlitzer import (
    pipes,  # pyright: ignore[reportMissingImports]  # pip install wurlitzer
)


def opt_result_diff(result, result_ref, rtol=1e-5, atol=1e-8):
    comparison = {
        "both success": result.success == result_ref.success,
        "both status": result.status == result_ref.status,
        "both fun": np.allclose(result.fun, result_ref.fun, rtol=rtol, atol=atol),
        "both x": np.allclose(result.x, result_ref.x, rtol=rtol, atol=atol),
        "both nit": result.nit == result_ref.nit,
        "both g": np.allclose(
            result.info["g"], result_ref.info["g"], rtol=rtol, atol=atol
        ),
        "both mult_g": np.allclose(
            result.info["mult_g"], result_ref.info["mult_g"], rtol=rtol, atol=atol
        ),
        "both mult_x_L": np.allclose(
            result.info["mult_x_L"], result_ref.info["mult_x_L"], rtol=rtol, atol=atol
        ),
        "both mult_x_U": np.allclose(
            result.info["mult_x_U"], result_ref.info["mult_x_U"], rtol=rtol, atol=atol
        ),
        "both nfev": result.nfev == result_ref.nfev,
        "both njev": result.njev == result_ref.njev,
        "both message": result.message == result_ref.message,
    }

    output = ""
    for key, value in comparison.items():
        if not value:
            test_name = key.split(" ")[1]
            output += (
                f"{test_name} test is False, therefore pycutest and our "
                f"implementation had different result {test_name} values\n"
            )
            if (
                test_name not in result.info.keys()
            ):  # ["g, mult_g", "mult_x_L", "mult_x_U"]:
                output += f"pycutest {test_name}: {getattr(result_ref, test_name)}\n"
                output += f"ours {test_name}: {getattr(result, test_name)}\n"
            else:
                output += f"pycutest {test_name}: {result_ref.info[test_name]}\n"
                output += f"ours {test_name}: {result.info[test_name]}\n"

    # see if all tests passed
    tests_passed = all(value for value in comparison.values())

    return tests_passed, output


@pytest.mark.parametrize("problem", sif2jax.problems)
def e2e_dense_ipopt_test(problem):
    """
    Run the pycutest problem and the sif2jax problem through cyipopts dense
    ipopt interface to conduct an end-to-end test. The resulting output,
    however suboptimal should be the same for both problems.
    """

    # Import the pycutest problem
    p_ref = pycutest.import_problem(problem.name)
    p = problem

    # Define the objective function, gradient, and hessian for both problems
    obj_ref = p_ref.obj
    obj_grad_ref = p_ref.grad
    obj_hess_ref = p_ref.ihess

    obj = lambda x: p.objective(x, p.args())
    obj_grad = jax.jacobian(obj)
    obj_hess = jax.hessian(obj)

    # Set tolerance and maximum iterations
    maxtol_coeff = 10
    maxiter = 100

    for i in range(maxtol_coeff):
        tol = 10 ** (-i)
        print(f"Running cyipopt with tol = {tol}")
        buf = io.StringIO()
        with pipes(stdout=buf, stderr=buf):
            result = cyipopt.minimize_ipopt(
                fun=obj,
                x0=p.y0(),
                jac=obj_grad,
                hess=obj_hess,
                tol=tol,
                options={"disp": 5, "maxiter": maxiter},
            )
            result_ref = cyipopt.minimize_ipopt(
                fun=obj_ref,
                x0=p_ref.x0,
                jac=obj_grad_ref,
                hess=obj_hess_ref,
                tol=tol,
                options={"disp": 5, "maxiter": maxiter},
            )
        ipopt_stdout = buf.getvalue()

        # assert that all numerical values in both scipy results objects are equal
        tests_passed, output = opt_result_diff(result, result_ref)
        assert tests_passed, (
            output
            + "\n\n\n below is our problems IPOPT output followed by "
            + "pycutest problems output\n\n\n"
            + ipopt_stdout
        )

        if result.success is False and i > 0:
            print(
                f"Highest tol with successful optimization in {maxiter} "
                f"iterations: {10 ** (-i + 1)}"
            )
            break
        if result.success is False and i == 0:
            print(f"Failed to converge with tol = {tol}")
            break

    else:
        print(f"Converged with maximum tol: {tol}")


if __name__ == "__main__":
    for problem in sif2jax.problems:
        print(f"Running end-to-end test for problem: {problem.name}")
        e2e_dense_ipopt_test(problem)

    name = "AKIVA"
    p_ref = pycutest.import_problem(name)
    p = sif2jax.problems[0]

    obj_ref = p_ref.obj
    obj_grad_ref = p_ref.grad
    obj_hess_ref = p_ref.ihess

    obj = lambda x: p.objective(x, p.args())
    obj_grad = jax.jacobian(obj)
    obj_hess = jax.hessian(obj)

    maxtol_coeff = 10
    maxiter = 100

    for i in range(maxtol_coeff):
        tol = 10 ** (-i)
        print(f"Running cyipopt with tol = {tol}")
        buf = io.StringIO()
        with pipes(stdout=buf, stderr=buf):
            result = cyipopt.minimize_ipopt(
                fun=obj,
                x0=p.y0(),
                jac=obj_grad,
                hess=obj_hess,
                tol=tol,
                options={"disp": 5, "maxiter": maxiter},
            )
            result_ref = cyipopt.minimize_ipopt(
                fun=obj_ref,
                x0=p_ref.x0,
                jac=obj_grad_ref,
                hess=obj_hess_ref,
                tol=tol,
                options={"disp": 5, "maxiter": maxiter},
            )
        ipopt_stdout = buf.getvalue()

        # assert that all numerical values in both scipy results objects are equal
        tests_passed, output = opt_result_diff(result, result_ref)
        assert tests_passed, (
            output
            + "\n\n\n below is our problems IPOPT output followed by "
            + "pycutest problems output\n\n\n"
            + ipopt_stdout
        )

        if result.success is False and i > 0:
            print(
                f"Highest tol with successful optimization in {maxiter} "
                f"iterations: {10 ** (-i + 1)}"
            )
            break
        if result.success is False and i == 0:
            print(f"Failed to converge with tol = {tol}")
            break

    else:
        print(f"Converged with maximum tol: {tol}")

    print("fin")
