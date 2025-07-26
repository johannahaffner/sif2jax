import equinox as eqx
import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed
# Attempts made: Fixed dtype issues, analyzed Hessian computation
# Suspected issues: Cross-derivative H[2,3] differs significantly from pycutest
# Additional resources needed: Verify the exact Hessian computation for scaled L2 groups
class HIMMELBF(AbstractUnconstrainedMinimisation):
    """Himmelblau 4 variable data fitting problem.

    A 4 variables data fitting problems by Himmelblau.

    Source: problem 32 in
    D.H. Himmelblau,
    "Applied nonlinear programming",
    McGraw-Hill, New-York, 1972.

    See Buckley#76 (p. 66)

    SIF input: Ph. Toint, Dec 1989.

    classification SUR2-AN-4-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem data
    a_data: tuple[float, ...] = eqx.field(
        default=(0.0, 0.000428, 0.001000, 0.001610, 0.002090, 0.003480, 0.005250),
        init=False,
    )
    b_data: tuple[float, ...] = eqx.field(
        default=(7.391, 11.18, 16.44, 16.20, 22.20, 24.02, 31.32), init=False
    )

    def objective(self, y, args):
        """Compute the objective function (vectorized).

        The objective is a sum of squared ratios.
        """
        x1, x2, x3, x4 = y

        # Convert data to JAX arrays for vectorized operations
        a_vals = jnp.array(self.a_data, dtype=y.dtype)
        b_vals = jnp.array(self.b_data, dtype=y.dtype)

        # Vectorized element function HF
        u = x1 * x1 + a_vals * x2 * x2 + a_vals * a_vals * x3 * x3
        v = b_vals * (1.0 + a_vals * x4 * x4)
        f_i = u / v

        # Vectorized group evaluation with constant 1.0 and scale 0.0001
        g_i = (f_i - 1.0) ** 2

        # Sum of all groups with scale (SCALE in SIF means multiply by 1/scale)
        return jnp.sum(g_i) / 0.0001

    @property
    def y0(self):
        return jnp.array([2.7, 90.0, 1500.0, 10.0])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        return jnp.array(318.572)
