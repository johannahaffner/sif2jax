import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class SROSENBR(AbstractUnconstrainedMinimisation):
    """The separable extension of Rosenbrock's function.

    Source: problem 21 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    SIF input: Ph. Toint, Dec 1989.
               added 2nd (correct) starting point, Dec 2024

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    n: int = 5000  # Number of variables (must be even)
    alpha: float = 100.0  # Penalty parameter

    def __init__(self, n: int = 5000):
        assert n % 2 == 0, "n must be even"
        self.n = n

    def objective(self, y, args):
        del args

        # Split variables into odd and even indices
        # x_{2i-1} corresponds to y[2*i-2] for i=1..n/2
        # x_{2i} corresponds to y[2*i-1] for i=1..n/2
        x_odd = y[0::2]  # x_1, x_3, x_5, ...
        x_even = y[1::2]  # x_2, x_4, x_6, ...

        # Standard extended Rosenbrock formulation
        # (as used by pycutest, despite what the SIF file says about scaling)
        # GA(i) = alpha * (x_{2i} - x_{2i-1}^2)^2
        # GB(i) = (1 - x_{2i-1})^2
        ga = self.alpha * (x_even - x_odd**2) ** 2
        gb = (1.0 - x_odd) ** 2

        return jnp.sum(ga) + jnp.sum(gb)

    @property
    def y0(self):
        # Match pycutest behavior exactly: only first two elements non-zero
        y = jnp.zeros(self.n)

        if self.y0_iD == 0:
            # First starting point - matches pycutest output
            y = y.at[0].set(1.2)  # x1 = 1.2 (positive)
            y = y.at[1].set(1.0)  # x2 = 1.0
            # Rest remain zeros
        else:
            # Second starting point (SROSENBR2) - use SIF pattern
            y = y.at[0::2].set(-1.2)  # odd indices
            y = y.at[1::2].set(1.0)  # even indices

        return y

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Solution: all variables equal to 1.0
        return jnp.ones(self.n)

    @property
    def expected_objective_value(self):
        return jnp.array(0.0)
