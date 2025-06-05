import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class DENSCHNA(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem A.

    This is a 2-dimensional unconstrained optimization problem with
    nonlinear terms including exponentials.

    Source: Problem from "Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations" by J.E. Dennis and R.B. Schnabel, 1983.

    Classification: OUR2-AN-2-0
    """

    n: int = 2  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # From AMPL model: x[1]^4 + (x[1]+x[2])^2 + (-1.0+exp(x[2]))^2
        term1 = x1**4
        term2 = (x1 + x2) ** 2
        term3 = (-1.0 + jnp.exp(x2)) ** 2

        return term1 + term2 + term3

    def y0(self):
        # Initial values based on problem specification
        return jnp.array([1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at x = [0.0, 0.0]
        return jnp.array([0.0, 0.0])

    def expected_objective_value(self):
        # At x = [0.0, 0.0]: 0^4 + (0+0)^2 + (-1+exp(0))^2 = 0 + 0 + 0 = 0
        return jnp.array(0.0)


# TODO: human review required
class DENSCHNB(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem B.

    This is a 2-dimensional unconstrained optimization problem with
    a product term (x1 - 2.0) * x2.

    Source: Problem from "Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations" by J.E. Dennis and R.B. Schnabel, 1983.

    Classification: SUR2-AN-2-0
    """

    n: int = 2  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # From AMPL model: (x[1]-2.0)^2 + ((x[1]-2.0)*x[2])^2 + (x[2]+1.0)^2
        term1 = (x1 - 2.0) ** 2
        term2 = ((x1 - 2.0) * x2) ** 2
        term3 = (x2 + 1.0) ** 2

        return term1 + term2 + term3

    def y0(self):
        # Initial values based on problem specification
        return jnp.array([1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # Based on the problem formulation, the minimum is at:
        return jnp.array([2.0, -1.0])

    def expected_objective_value(self):
        # At x = [2.0, -1.0]: (2-2)^2 + ((2-2)*(-1))^2 + (-1+1)^2 = 0
        return jnp.array(0.0)


# TODO: human review required
class DENSCHNC(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem C.

    This is a 2-dimensional unconstrained optimization problem with
    squares of variables and an exponential term.

    Source: Problem from "Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations" by J.E. Dennis and R.B. Schnabel, 1983.

    Classification: SUR2-AN-2-0
    """

    n: int = 2  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # From AMPL model: (-2+x[1]^2+x[2]^2)^2 + (-2+exp(x[1]-1)+x[2]^3)^2
        term1 = (-2.0 + x1**2 + x2**2) ** 2
        term2 = (-2.0 + jnp.exp(x1 - 1.0) + x2**3) ** 2

        return term1 + term2

    def y0(self):
        # Initial values from AMPL model: x[1]=2, x[2]=3
        return jnp.array([2.0, 3.0])

    def args(self):
        return None

    def expected_result(self):
        # Solution depends on solving the system of equations
        return None

    def expected_objective_value(self):
        # The minimum value is 0 when both terms equal 0
        return jnp.array(0.0)


# TODO: human review required
class DENSCHND(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem D.

    This is a 3-dimensional unconstrained optimization problem with
    polynomial terms up to the fourth power.

    Source: Problem from "Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations" by J.E. Dennis and R.B. Schnabel, 1983.

    Classification: SUR2-AN-3-0
    """

    n: int = 3  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # From AMPL model:
        # (x[1]^2+x[2]^3-x[3]^4)^2 + (2*x[1]*x[2]*x[3])^2 +
        # (2*x[1]*x[2]-3*x[2]*x[3]+x[1]*x[3])^2
        term1 = (x1**2 + x2**3 - x3**4) ** 2
        term2 = (2.0 * x1 * x2 * x3) ** 2
        term3 = (2.0 * x1 * x2 - 3.0 * x2 * x3 + x1 * x3) ** 2

        return term1 + term2 + term3

    def y0(self):
        # Initial values based on problem specification
        return jnp.array([10.0, 10.0, 10.0])

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.array([0.0, 0.0, 0.0])

    def expected_objective_value(self):
        # At the origin, all terms evaluate to 0
        return jnp.array(0.0)


# TODO: human review required
class DENSCHNE(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem E.

    This is a 3-dimensional unconstrained optimization problem with
    squares of variables and an exponential term.

    Source: Problem from "Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations" by J.E. Dennis and R.B. Schnabel, 1983.

    Classification: SUR2-AN-3-0
    """

    n: int = 3  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # From AMPL model: x[1]^2 + (x[2]+x[2]^2)^2 + (-1+exp(x[3]))^2
        term1 = x1**2
        term2 = (x2 + x2**2) ** 2
        term3 = (-1.0 + jnp.exp(x3)) ** 2

        return term1 + term2 + term3

    def y0(self):
        # Initial values based on problem specification
        return jnp.array([2.0, 3.0, -8.0])

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at x1=0, x2=0, x3=0 (where exp(0)=1)
        return jnp.array([0.0, 0.0, 0.0])

    def expected_objective_value(self):
        # At x = [0, 0, 0]: 0^2 + (0+0^2)^2 + (-1+exp(0))^2 = 0 + 0 + 0 = 0
        return jnp.array(0.0)


# TODO: human review required
class DENSCHNF(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem F.

    This is a 2-dimensional unconstrained optimization problem with
    a sum of squares formulation.

    Source: Problem from "Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations" by J.E. Dennis and R.B. Schnabel, 1983.

    Classification: SUR2-AY-2-0
    """

    n: int = 2  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # From AMPL model:
        # (2*(x1+x2)^2+(x1-x2)^2-8)^2 + (5*x1^2+(x2-3)^2-9)^2
        term1 = (2.0 * (x1 + x2) ** 2 + (x1 - x2) ** 2 - 8.0) ** 2
        term2 = (5.0 * x1**2 + (x2 - 3.0) ** 2 - 9.0) ** 2

        return term1 + term2

    def y0(self):
        # Initial values based on problem specification
        return jnp.array([2.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        # Solution depends on solving the system of equations
        return None

    def expected_objective_value(self):
        # The minimum value is 0 when both terms equal 0
        return jnp.array(0.0)
