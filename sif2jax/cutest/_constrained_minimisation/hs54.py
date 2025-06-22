import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS54(AbstractConstrainedMinimisation):
    """Problem 54 from the Hock-Schittkowski test collection.

    A 6-variable nonlinear objective function with one linear equality constraint
    and bounds on variables.

    f(x) = -exp(-h(x)/2)

    where h(x) = ((x₁ - 1.E6)²/6.4E7 + (x₁ - 1.E4)(x₂ - 1)/2.E4
                  + (x₂ - 1)²)(x₃ - 2.E6)²/(.9614.9E13)
                  + (x₄ - 10)²/2.5E3 + (x₅ - 1.E-3)²/2.5E-3
                  + (x₆ - 1.E8)²/2.5E17

    Subject to:
        x₁ + 4.E3x₂ - 1.76E4 = 0
        0 ≤ x₁ ≤ 2.E4, -10 ≤ x₂ ≤ 10, 0 ≤ x₃ ≤ 1.E7
        0 ≤ x₄ ≤ 20, -1 ≤ x₅ ≤ 1, 0 ≤ x₆ ≤ 2.E8

    Source: problem 54 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8], Picket [50]

    Classification: GLR-T1-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5, x6 = y
        # Based on SIF file structure
        rho = 0.2
        factor = 1.0 / (1 - rho**2)  # = 1/0.96 = 1.04166...
        mu = jnp.array([1.0e4, 1.0, 2.0e6, 1.0e1, 1.0e-3, 1.0e8])
        sigma = jnp.array([8.0e3, 1.0, 7.0e6, 5.0e1, 5.0e-2, 5.0e8])

        # Elements E1-E6: ((xi - mui)/sigmai)^2
        e1 = ((x1 - mu[0]) / sigma[0]) ** 2
        e2 = ((x2 - mu[1]) / sigma[1]) ** 2
        e3 = ((x3 - mu[2]) / sigma[2]) ** 2
        e4 = ((x4 - mu[3]) / sigma[3]) ** 2
        e5 = ((x5 - mu[4]) / sigma[4]) ** 2
        e6 = ((x6 - mu[5]) / sigma[5]) ** 2

        # Element F1: 2*rho*(x1-mu1)/sigma1 * (x2-mu2)/sigma2
        f1 = 2 * rho * ((x1 - mu[0]) / sigma[0]) * ((x2 - mu[1]) / sigma[1])

        # Q = factor*(e1 + e2 + f1) + e3 + e4 + e5 + e6
        q = factor * (e1 + e2 + f1) + e3 + e4 + e5 + e6

        return -jnp.exp(-0.5 * q)

    def y0(self):
        return jnp.array([6.0e3, 1.5, 4.0e6, 2.0, 3.0e-3, 5.0e7])

    def args(self):
        return None

    def expected_result(self):
        # Solution approximated as rational numbers
        return jnp.array([91600.0 / 7.0, 179.0 / 70.0, 2.0e6, 10.0, 1.0e-3, 1.0e8])

    def expected_objective_value(self):
        return -jnp.exp(-27.0 / 280.0)

    def bounds(self):
        lower = jnp.array([0.0, -10.0, 0.0, 0.0, -1.0, 0.0])
        upper = jnp.array([2.0e4, 10.0, 1.0e7, 20.0, 1.0, 2.0e8])
        return (lower, upper)

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6 = y
        # Equality constraint
        eq = x1 + 4.0e3 * x2 - 1.76e4
        equality_constraints = jnp.array([eq])
        return equality_constraints, None
