import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed
# Attempts made: Same formulation issues as HS74 (constraints differ from pycutest)
# Suspected issues: Constraint values and Jacobians differ significantly from pycutest
# Resources needed: Access to original Hock-Schittkowski formulation or pycutest source
class HS75(AbstractConstrainedMinimisation):
    """Problem 75 from the Hock-Schittkowski test collection.

    A 4-variable nonlinear objective function with two inequality constraints, three
    equality constraints and bounds.

    f(x) = 3*x₁ + 1.E-6*x₁³ + 2*x₂ + (2/3)*E-6*x₂³

    Subject to:
        x₄ - x₃ + a₂ ≥ 0
        x₃ - x₄ + a₂ ≥ 0
        -x₁ - 894.8 - 1000*sin(x₃ + 0.25) - 1000*sin(x₄ + 0.25) = 0
        -x₂ - 894.8 + 1000*sin(x₃ - 0.25) + 1000*sin(x₃ - x₄ - 0.25) = 0
        -1294.8 + 1000*sin(x₄ - 0.25) + 1000*sin(x₄ - x₃ - 0.25) = 0
        0 ≤ xᵢ ≤ 1200, i=1,2
        -a₂ ≤ xᵢ ≤ a₂, i=3,4

    where a₂ = 0.48

    Source: problem 75 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Beuneu [9]

    Classification: PGR-P1-(1,2)
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4 = y
        return (
            3 * x1
            + 1.0e-6 * x1 * x1 * x1
            + 2 * x2
            + (2.0 / 3.0) * 1.0e-6 * x2 * x2 * x2
        )

    @property
    def y0(self):
        return jnp.array([0.0, 0.0, 0.0, 0.0])  # not feasible according to the problem

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return jnp.array([776.1592, 925.1949, 0.05110879, -0.4288911])

    @property
    def expected_objective_value(self):
        return jnp.array(5174.4129)

    @property
    def bounds(self):
        # Problem parameter
        a2 = 0.48
        return (jnp.array([0.0, 0.0, -a2, -a2]), jnp.array([1200.0, 1200.0, a2, a2]))

    def constraint(self, y):
        x1, x2, x3, x4 = y
        # Problem parameter
        a2 = 0.48

        # Inequality constraints (g(x) ≥ 0)
        ineq1 = x4 - x3 + a2
        ineq2 = x3 - x4 + a2

        # Precompute common trigonometric values for efficiency
        sin_x3_025 = jnp.sin(x3 - 0.25)
        sin_x4_025 = jnp.sin(x4 - 0.25)
        sin_x3_x4_025 = jnp.sin(x3 - x4 - 0.25)
        sin_x4_x3_025 = jnp.sin(x4 - x3 - 0.25)

        # Equality constraints (from SIF file)
        # C3: -X1 - 894.8 - 1000*sin(X3+0.25) - 1000*sin(X4+0.25) = 0
        # C4: -X2 - 894.8 + 1000*sin(X3-0.25) + 1000*sin((X3-X4)-0.25) = 0
        # C5: -1294.8 + 1000*sin(X4-0.25) + 1000*sin((X4-X3)-0.25) = 0
        eq1 = -x1 - 894.8 - 1000.0 * jnp.sin(x3 + 0.25) - 1000.0 * jnp.sin(x4 + 0.25)
        eq2 = -x2 - 894.8 + 1000.0 * sin_x3_025 + 1000.0 * sin_x3_x4_025
        eq3 = -1294.8 + 1000.0 * sin_x4_025 + 1000.0 * sin_x4_x3_025

        equality_constraints = jnp.array([eq1, eq2, eq3])
        inequality_constraints = jnp.array([ineq1, ineq2])
        return equality_constraints, inequality_constraints
