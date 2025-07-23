import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class DEVGLA2NE(AbstractNonlinearEquations):
    """
    SCIPY global optimization benchmark example DeVilliersGlasser02

    Fit: y  = x_1 x_2^t tanh ( t x_3 + sin( t x_4 ) ) cos( t e^x_5 )  +  e

    Source:  Problem from the SCIPY benchmark set
        https://github.com/scipy/scipy/tree/master/benchmarks/ ...
                benchmarks/go_benchmark_functions

    Nonlinear-equation formulation of DEVGLA2.SIF

    SIF input: Nick Gould, Jan 2020

    classification NOR2-MN-5-16
    """

    m: int = 16
    n: int = 5
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        return jnp.array([20.0, 2.0, 2.0, 2.0, 0.2], dtype=jnp.float64)

    def num_residuals(self) -> int:
        return self.m

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the DeVilliersGlasser02 problem"""
        x1, x2, x3, x4, x5 = y[0], y[1], y[2], y[3], y[4]

        # Precompute data values
        a = 1.27
        residuals = jnp.zeros(self.m, dtype=jnp.float64)

        for i in range(self.m):
            t = i * 0.1  # t = (i-1) * 0.1 in 0-based indexing
            at = a**t
            tp = t * 3.012
            tp2 = t * 2.13
            stp2 = jnp.sin(tp2)
            tpa = tp + stp2
            htpa = jnp.tanh(tpa)
            ec = jnp.exp(0.507)
            ect = ec * t
            cect = jnp.cos(ect)
            p = at * htpa
            pp = p * cect
            ppp = pp * 53.81
            y_i = ppp

            # Element DG2: x1 * x2^t * tanh(t * x3 + sin(t * x4)) * cos(t * exp(x5))
            x2t = x2**t
            x3t = x3 * t
            x4t = x4 * t
            sinx4t = jnp.sin(x4t)
            a_val = x3t + sinx4t
            f3 = jnp.tanh(a_val)
            ex5 = jnp.exp(x5)
            tex5 = t * ex5
            ctex5 = jnp.cos(tex5)
            f4 = ctex5

            f_i = x1 * x2t * f3 * f4
            residuals = residuals.at[i].set(f_i - y_i)

        return residuals

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # Solution from SIF file
        return jnp.array([53.81, 1.27, 3.012, 2.13, 0.507], dtype=jnp.float64)

    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """No bounds for this problem."""
        return None
