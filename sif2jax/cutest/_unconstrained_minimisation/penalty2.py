import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class PENALTY2(AbstractUnconstrainedMinimisation):
    """
    PENALTY2 problem.

    The second penalty function

    This is a nonlinear least-squares problem with M=2*N groups.
     Group 1 is linear.
     Groups 2 to N use 2 nonlinear elements.
     Groups N+1 to M-1 use 1 nonlinear element.
     Group M uses N nonlinear elements.
    The Hessian matrix is dense.

    Source:  Problem 24 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#112 (p. 80)

    SIF input: Ph. Toint, Dec 1989.

    classification SUR2-AN-V-0

    TODO: Human review needed
    Attempts made: Multiple interpretations of SCALE factor in SIF
    Suspected issues: Incorrect understanding of how SCALE interacts with L2 groups
    The objective/gradient values are off by a factor of ~10^10
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 200

    def objective(self, y, args):
        del args
        n = self.n

        # Parameters
        a = 0.00001
        b = 1.0

        # Group 1: (x[0] - 0.2)^2 scaled by 1/b
        # Note: SCALE in SIF might multiply the squared value
        g1 = (y[0] - 0.2) ** 2 / b

        # Precompute exp(0.1*x) for all elements
        exp_x = jnp.exp(0.1 * y)

        # Groups 2 to n: 1/a * (exp(0.1*x[i]) + exp(0.1*x[i-1]) -
        #                        (exp(0.1*i) + exp(0.1*(i-1))))^2
        if n > 1:
            # Vectorized computation
            i_vals = jnp.arange(2, n + 1)  # i from 2 to n
            y_vals = jnp.exp(0.1 * i_vals) + jnp.exp(0.1 * (i_vals - 1))
            a_vals = exp_x[1:n]  # exp(0.1*x[i]) for i=2 to n (0-based: 1 to n-1)
            b_vals = exp_x[
                0 : n - 1
            ]  # exp(0.1*x[i-1]) for i=2 to n (0-based: 0 to n-2)

            group_vals = a_vals + b_vals - y_vals
            groups_2_to_n = (1.0 / a) * jnp.sum(group_vals**2)
        else:
            groups_2_to_n = 0.0

        # Groups n+1 to m-1: 1/a * (exp(0.1*x[i-n+1]) - exp(-0.1))^2
        if n > 1:
            # i from n+1 to 2n-1, so i-n+1 from 2 to n (0-based: 1 to n-1)
            c_vals = exp_x[1:n]  # exp(0.1*x[j]) for j=2 to n
            em1_10 = jnp.exp(-0.1)
            group_vals = c_vals - em1_10
            groups_np1_to_m1 = (1.0 / a) * jnp.sum(group_vals**2)
        else:
            groups_np1_to_m1 = 0.0

        # Group m: 1/b * (sum_{j=1}^n (n-j+1)*x[j]^2 - 1)^2
        # Weights: n, n-1, ..., 1
        weights = jnp.arange(n, 0, -1)
        weighted_sum = jnp.sum(weights * y**2)
        gm = (1.0 / b) * (weighted_sum - 1.0) ** 2

        return g1 + groups_2_to_n + groups_np1_to_m1 + gm

    def y0(self):
        # Standard starting point (0.5, ..., 0.5)
        return inexact_asarray(0.5 * jnp.ones(self.n))

    def args(self):
        return None

    def expected_result(self):
        # Solution not provided in detail in SIF file
        return None

    def expected_objective_value(self):
        # Solution values provided for some values of n
        n = self.n
        if n == 4:
            return jnp.array(9.37629e-6)
        elif n == 10:
            return jnp.array(2.93660e-4)
        elif n == 50:
            return jnp.array(4.29609813)
        elif n == 100:
            return jnp.array(97096.0840)
        else:
            # For n=200 and others, solution not provided
            return None

    def num_variables(self):
        return self.n
