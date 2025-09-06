import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# Module-level private constants for MANNE problem
def _compute_manne_parameters():
    """Compute derived parameters following SIF logic."""
    import math

    # Constants
    _t = 2000
    _grow = 0.03
    _beta = 0.95
    _xk0 = 3.0
    _xc0 = 0.95
    _xi0 = 0.05
    _b = 0.25

    # A = (XC0 + XI0) / (XK0^B)
    _xk0_b = _xk0**_b
    _a = (_xc0 + _xi0) / _xk0_b

    # Growth factor computation
    _one_minus_b = 1.0 - _b
    _one_plus_g = 1.0 + _grow
    _log_one_plus_g = math.log(_one_plus_g)
    _some = _log_one_plus_g * _one_minus_b
    _gfac = math.exp(_some)

    # AT(i) = A * GFAC^(i-1) for i = 1..T
    _i_values = jnp.arange(1, _t + 1, dtype=jnp.float64)
    _at = _a * (_gfac ** (_i_values - 1))

    # BT(i) = BETA^(i-1) for i = 1..T-1, BT(T) = BETA^(T-1) / (1-BETA)
    _bt_values = _beta ** jnp.arange(_t, dtype=jnp.float64)
    _bt_values = _bt_values.at[-1].set(_bt_values[-1] / (1.0 - _beta))
    _bt = _bt_values

    return _a, _gfac, _at, _bt


# Pre-compute all derived constants as module-level private variables
_a, _gfac, _at, _bt = _compute_manne_parameters()


# TODO: Human review needed
# Attempts made: [vectorized implementation, dtype fixes, constraint structure analysis]
# Suspected issues: [objective function sign/structure,
#                    constraint dimension mismatches,
#                    complex econometric model interpretation]
# Resources needed: [detailed SIF econometric model analysis,
#                    constraint structure verification]


class MANNE(AbstractConstrainedMinimisation):
    """MANNE - A variable dimension econometric equilibrium problem.

    T = 2000 periods, n = 6000 variables (3*T).

    Variables: C(i), I(i), K(i) for i = 1..T (consumptions, investments, capitals)

    Objective: maximize sum of BT(i) * log(C(i)) for i = 1..T
    (converted to minimize negative sum)

    Constraints:
    - Nonlinear: C(i) + I(i) ≤ AT(i) * K(i)^B for i = 1..T
    - Linear: K(i+1) = K(i) + I(i) for i = 1..T-1
    - Terminal: K(T) * GROW ≤ I(T)

    Start: K(1) = 3.05, K(i) = 3.0 + 0.1*(i-1) for i ≥ 2, C(i) = 0.95, I(i) = 0.05

    Source: B. Murtagh and M. Saunders,
    Mathematical Programming Studies 16, pp. 84-117, (example 5.12).

    SIF input: N. Gould and Ph. Toint, March 1990.

    Classification: OOR2-MN-V-V (T = 2000)
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem constants (T = 2000)
    T: int = 2000
    GROW: float = 0.03
    BETA: float = 0.95
    XK0: float = 3.0
    XC0: float = 0.95
    XI0: float = 0.05
    B: float = 0.25

    def objective(self, y, args):
        del args
        # Variables are ordered: [C(1), I(1), K(1), C(2), I(2), K(2), ...]
        c_values = y[::3]  # C(1), C(2), ..., C(T)

        # Objective: maximize sum(BT(i) * log(C(i))) = minimize -sum(BT(i) * log(C(i)))
        log_c = jnp.log(c_values)
        return -jnp.sum(_bt * log_c)

    @property
    def y0(self):
        # Starting point following SIF specification
        c_start = jnp.full(self.T, self.XC0)  # C(i) = 0.95
        i_start = jnp.full(self.T, self.XI0)  # I(i) = 0.05

        # K(1) = 3.05, K(i) = 3.0 + 0.1*(i-1) for i ≥ 2
        k_start = jnp.zeros(self.T)
        k_start = k_start.at[0].set(3.05)
        k_indices = jnp.arange(1, self.T, dtype=jnp.float64)
        k_start = k_start.at[1:].set(3.0 + 0.1 * k_indices)

        # Interleave variables: [C(1), I(1), K(1), C(2), I(2), K(2), ...]
        y0_array = jnp.zeros(3 * self.T)
        y0_array = y0_array.at[::3].set(c_start)  # C values
        y0_array = y0_array.at[1::3].set(i_start)  # I values
        y0_array = y0_array.at[2::3].set(k_start)  # K values

        return y0_array

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        # From SIF file comment: solution approximately -0.97457259
        return jnp.array(-0.97457259)

    @property
    def bounds(self):
        """Bounds: K(1) = 3.05, K(i) ≥ 3.05, C(i) ≥ 0.95, 0.05 ≤ I(i) ≤ large_value."""
        lower_bounds = jnp.full(3 * self.T, -jnp.inf)
        upper_bounds = jnp.full(3 * self.T, jnp.inf)

        # C(i) ≥ 0.95
        lower_bounds = lower_bounds.at[::3].set(0.95)

        # I(i) ≥ 0.05, I(i) ≤ large upper bound from SIF
        lower_bounds = lower_bounds.at[1::3].set(0.05)
        large_upper = (1.04 ** float(self.T)) * 0.05  # Upper bound calculation from SIF
        upper_bounds = upper_bounds.at[1::3].set(large_upper)

        # K(1) = 3.05 (fixed), K(i) ≥ 3.05 for i ≥ 2
        lower_bounds = lower_bounds.at[2].set(3.05)  # K(1) fixed
        upper_bounds = upper_bounds.at[2].set(3.05)  # K(1) fixed
        k_indices = jnp.arange(1, self.T) * 3 + 2  # K(2), K(3), ..., K(T) indices
        lower_bounds = lower_bounds.at[k_indices].set(3.05)

        return (lower_bounds, upper_bounds)

    def constraint(self, y):
        """
        Constraints:
        1. Nonlinear: C(i) + I(i) ≤ AT(i) * K(i)^B for i = 1..T (inequality)
        2. Linear: K(i+1) = K(i) + I(i) for i = 1..T-1 (equality)
        3. Terminal: K(T) * GROW ≤ I(T) (inequality)
        """
        # Extract variables
        c_values = y[::3]  # C(1), C(2), ..., C(T)
        i_values = y[1::3]  # I(1), I(2), ..., I(T)
        k_values = y[2::3]  # K(1), K(2), ..., K(T)

        # Equality constraints: K(i+1) = K(i) + I(i) for i = 1..T-1
        k_next = k_values[1:]  # K(2), K(3), ..., K(T)
        k_curr = k_values[:-1]  # K(1), K(2), ..., K(T-1)
        i_curr = i_values[:-1]  # I(1), I(2), ..., I(T-1)
        equality_constraints = k_next - k_curr - i_curr  # Should be 0

        # Inequality constraints
        inequality_list = []

        # Nonlinear: AT(i) * K(i)^B - C(i) - I(i) ≥ 0 for i = 1..T
        k_power = k_values**self.B
        production = _at * k_power
        consumption = c_values + i_values
        nonlinear_ineq = production - consumption
        inequality_list.append(nonlinear_ineq)

        # Terminal constraint: I(T) - K(T) * GROW ≥ 0
        terminal_ineq = i_values[-1] - k_values[-1] * self.GROW
        inequality_list.append(jnp.array([terminal_ineq]))

        inequality_constraints = jnp.concatenate(inequality_list)

        return equality_constraints, inequality_constraints
