from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..._problem import AbstractConstrainedMinimisation


# Load data at module level
_data_file = Path(__file__).parent / "data" / "portfl4_data.npz"
if _data_file.exists():
    _data = np.load(_data_file)
    _F = jnp.array(_data["F"])  # 12 x 62 factor matrix
    _R = jnp.array(_data["R"])  # 62 returns vector
else:
    # Fallback: empty arrays for testing
    _F = jnp.zeros((12, 62))
    _R = jnp.zeros(62)


class PORTFL4(AbstractConstrainedMinimisation):
    """Portfolio optimization problem PORTFL4.

    Establish the sensitivity of certain obligations combinations in portfolio
    analysis, based on the methodology of W. Sharpe.

    The problem minimizes the squared error between portfolio returns and
    factor-based predictions:

    minimize: sum_i (R_i - sum_j F_{j,i} * S_j)^2

    subject to: sum_j S_j = 1
                0 <= S_j <= 1 for all j

    where:
    - S_j are the portfolio weights (12 variables)
    - F_{j,i} is the factor return matrix (12 factors x 62 observations)
    - R_i are the realized portfolio returns (62 observations)

    Source:
    DATASTREAM Period: 15.1.91 to 15.3.96 (62 observations),
    collected by D. Baudoux, July 1996.

    SIF input: Ph. Toint, July 1996.

    Classification: SLR2-MN-12-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n_var(self):
        return jnp.array(12)

    @property
    def n_con(self):
        return jnp.array(1)

    @property
    def y0(self):
        # Start point: uniform distribution (1/12 for each weight)
        return jnp.ones(12) / 12

    @property
    def args(self):
        return None

    @property
    def bounds(self):
        # Box constraints: 0 <= S_j <= 1
        lower = jnp.zeros(12)
        upper = jnp.ones(12)
        return lower, upper

    def objective(self, y, args):
        """Least squares objective: sum_i (R_i - sum_j F_{j,i} * S_j)^2"""
        # y contains the 12 portfolio weights S_j
        # Compute predicted returns: F^T @ y gives predictions for each observation
        predicted = jnp.dot(_F.T, y)  # (62,)
        # Compute squared errors
        errors = _R - predicted
        return jnp.sum(errors**2)

    def constraint(self, y):
        """Constraint: sum of weights equals 1"""
        # Equality constraint: sum(S_j) = 1
        eq_constraint = jnp.sum(y) - 1.0
        return jnp.array([eq_constraint]), None

    @property
    def expected_result(self):
        # Not available from SIF file
        return None

    @property
    def expected_objective_value(self):
        # From SIF file: *LO SOLTN 2.63069364D-2
        return jnp.array(2.63069364e-2)
