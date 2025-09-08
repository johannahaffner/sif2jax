"""Broyden tridiagonal system of nonlinear equations.

Source: problem 30 in
J.J. More', B.S. Garbow and K.E. Hillstrom,
"Testing Unconstrained Optimization Software",
ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

See also Toint#17 and Buckley#78.
SIF input: Ph. Toint, Dec 1989.

Classification: NQR2-AN-V-V

TODO: Human review needed - constraint values don't match pycutest
Our implementation matches the SIF file specification exactly, but
pycutest returns different values. This may be a pycutest issue
or a different interpretation of the NQR2 classification.
"""

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class BROYDN3D(AbstractNonlinearEquations):
    """Broyden tridiagonal system of nonlinear equations."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    KAPPA1: float = 2.0
    KAPPA2: float = 1.0

    @property
    def n(self):
        """Number of variables (default 5000)."""
        return jnp.array(5000)

    @property
    def y0(self):
        """Initial guess."""
        return jnp.full(self.n, -1.0)

    @property
    def args(self):
        """No additional arguments."""
        return None

    def constraint(self, y):
        """Compute the system of nonlinear equations.

        The equations are:
        E(1) = (3 - KAPPA1*x(1))*x(1) - KAPPA2 - 2*x(2)
        E(i) = (3 - KAPPA1*x(i))*x(i) - KAPPA2 - x(i-1) - 2*x(i+1) for i=2,...,n-1
        E(n) = (3 - KAPPA1*x(n))*x(n) - KAPPA2 - x(n-1)
        """
        # args not used

        n = self.n
        x = y

        # Element function: (3 - KAPPA1*x)*x
        def broy_element(xi):
            return (3.0 - self.KAPPA1 * xi) * xi

        # Apply element function to all variables
        broy_terms = jnp.array([broy_element(x[i]) for i in range(n)])

        # Initialize equations
        equations = jnp.zeros(n)

        # E(1) = broy(x1) - KAPPA2 - 2*x(2)
        equations = equations.at[0].set(broy_terms[0] - self.KAPPA2 - 2.0 * x[1])

        # E(i) = broy(xi) - KAPPA2 - x(i-1) - 2*x(i+1) for i=2,...,n-1
        for i in range(1, n - 1):
            equations = equations.at[i].set(
                broy_terms[i] - self.KAPPA2 - x[i - 1] - 2.0 * x[i + 1]
            )

        # E(n) = broy(xn) - KAPPA2 - x(n-1)
        equations = equations.at[n - 1].set(broy_terms[n - 1] - self.KAPPA2 - x[n - 2])

        return equations, None

    @property
    def bounds(self):
        """No bounds on variables."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value is 0.0 for exact solution."""
        return jnp.array(0.0)
