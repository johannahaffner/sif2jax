"""Base class for MISRA nonlinear equations problems."""

from abc import ABC, abstractmethod

from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class MISRABase(AbstractNonlinearEquations, ABC):
    """Base class for MISRA nonlinear equations problems.

    All MISRA problems are NIST data fitting problems given as inconsistent sets of
    nonlinear equations. They share the same experimental data but differ in their
    mathematical models.

    Source: Problem from the NIST nonlinear regression test set
      http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Misra, D., NIST (1978).
      Dental Research Monomolecular Adsorption Study.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification: NOR2-MN-2-14
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    @property
    def x_data(self) -> Float[Array, "14"]:
        """Experimental x data values."""
        return jnp.array(
            [
                77.6,
                114.9,
                141.1,
                190.8,
                239.9,
                289.0,
                332.8,
                378.4,
                434.8,
                477.3,
                536.8,
                593.1,
                689.1,
                760.0,
            ]
        )

    @property
    def y_data(self) -> Float[Array, "14"]:
        """Experimental y data values."""
        return jnp.array(
            [
                10.07,
                14.73,
                17.94,
                23.93,
                29.61,
                35.18,
                40.02,
                44.82,
                50.76,
                55.05,
                61.01,
                66.40,
                75.47,
                81.78,
            ]
        )

    @abstractmethod
    def model(
        self, b1: Float[Array, ""], b2: Float[Array, ""], x_data: Float[Array, "14"]
    ) -> Float[Array, "14"]:
        """Mathematical model function to be implemented by subclasses.

        Args:
            b1: First parameter
            b2: Second parameter
            x_data: Input data points

        Returns:
            Model predictions for the given parameters and data
        """
        pass

    def residual(self, y, args) -> Float[Array, "14"]:
        """Residual function for the nonlinear equations."""
        b1, b2 = y
        model_predictions = self.model(b1, b2, self.x_data)
        return model_predictions - self.y_data

    @property
    def y0(self) -> Float[Array, "2"]:
        """Initial guess for the optimization problem."""
        if self.y0_iD == 0:
            # START1
            return jnp.array([500.0, 0.0001])
        else:
            # START2
            return jnp.array([450.0, 0.0003])

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    @property
    def expected_result(self) -> None:
        """Expected result of the optimization problem."""
        # The SIF files don't provide solutions
        return None

    @property
    def expected_objective_value(self) -> Float[Array, ""]:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """No bounds for these problems."""
        return None
