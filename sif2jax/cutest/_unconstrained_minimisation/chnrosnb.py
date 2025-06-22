import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class CHNROSNB(AbstractUnconstrainedMinimisation):
    """The chained Rosenbrock function (Toint).

    This is a variant of the Rosenbrock function where each variable is connected to
    its neighbors in a chain-like structure. The variables in each pair (x_{i-1}, x_i)
    are scaled by a parameter alpha_i.

    Source:
    Ph.L. Toint,
    "Some numerical results using a sparse matrix updating formula in
    unconstrained optimization",
    Mathematics of Computation, vol. 32(114), pp. 839-852, 1978.

    See also Buckley#46 (n = 25) (p. 45).
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 50  # Number of variables (default 50, but can also be 10 or 25)

    def objective(self, y, args):
        del args

        # Alpha values from the SIF file
        alphas = jnp.array(
            [
                1.25,
                1.40,
                2.40,
                1.40,
                1.75,
                1.20,
                2.25,
                1.20,
                1.00,
                1.10,
                1.50,
                1.60,
                1.25,
                1.25,
                1.20,
                1.20,
                1.40,
                0.50,
                0.50,
                1.25,
                1.80,
                0.75,
                1.25,
                1.40,
                1.60,
                2.00,
                1.00,
                1.60,
                1.25,
                2.75,
                1.25,
                1.25,
                1.25,
                3.00,
                1.50,
                2.00,
                1.25,
                1.40,
                1.80,
                1.50,
                2.20,
                1.40,
                1.50,
                1.25,
                2.00,
                1.50,
                1.25,
                1.40,
                0.60,
                1.50,
            ]
        )

        # Ensure we're using the correct number of alpha values
        # based on problem dimension
        alphas = alphas[: self.n]

        # Based on AMPL model in chnrosnb.mod
        # sum {i in 2..n} (
        # (x[i-1]-x[i]^2)^2*16*alph[i]^2 +
        # (x[i]-1.0)^2
        # )

        # Vectorized computation for i in 2..n (0-based: 1..n-1)
        # Get corresponding alpha values
        alpha_vals = alphas[1 : self.n]  # alphas[1] to alphas[n-1]

        # Get x[i-1] and x[i] values
        x_i_minus_1 = y[:-1]  # y[0] to y[n-2]
        x_i = y[1:]  # y[1] to y[n-1]

        # Compute terms more efficiently
        x_i_sq = x_i * x_i
        diff = x_i_minus_1 - x_i_sq
        alpha_sq = alpha_vals * alpha_vals

        term1 = 16.0 * alpha_sq * (diff * diff)
        term2 = (x_i - 1.0) ** 2

        # Sum all terms
        return jnp.sum(term1 + term2)

    def y0(self):
        # Initial values from SIF file (all -1.0)
        return jnp.full(self.n, -1.0)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution has all components equal to 1
        return jnp.ones(self.n)

    def expected_objective_value(self):
        # According to the SIF file comment, the optimal objective value is 0.0
        return jnp.array(0.0)
