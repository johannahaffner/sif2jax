import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class DTOC1NB(AbstractConstrainedMinimisation):
    """A discrete time optimal control (DTOC) problem with nonlinear transitions.

    The system has N time periods, NX control variables and NY state variables.
    The nonlinearity parameter mu is set to 0.05.

    The problem is not convex.

    Sources: problem 1 in
    T.F. Coleman and A. Liao,
    "An Efficient Trust Region Method for Unconstrained Discrete-Time Optimal
    Control Problems",
    Tech. Report, ctc93tr144,  Advanced Computing Research Institute,
    Cornell University, 1992.

    L.Z. Liao and C.A. Shoemaker,
    "Advantages of differential dynamic programming over Newton's method for
    discrete-time optimal control problems",
    Tech. Report ctc92tr97, Advanced Computing Research Institute,
    Cornell University, 1992.

    SIF input: Ph. Toint, August 1993

    classification OQR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters - using the default values from SIF
    n_periods: int = 1000  # N
    n_controls: int = 2  # NX
    n_states: int = 4  # NY
    mu: float = 0.05  # Nonlinearity parameter

    # Total number of variables and constraints
    # Total variables include fixed Y(1,i) variables
    n: int = (n_periods - 1) * n_controls + n_periods * n_states  # 5998
    m: int = (n_periods - 1) * n_states  # 3996

    @property
    def y0(self):
        # All variables start at 0.0
        return jnp.zeros(self.n)

    @property
    def args(self):
        # Build the transition matrices B and C
        nx_ny = self.n_controls + self.n_states
        inv_nx_ny = 1.0 / nx_ny
        mu_over_nx_ny = self.mu * inv_nx_ny

        # B(i,j) = (i-j) / (NX+NY) for i=1..NY, j=1..NX
        i_indices = jnp.arange(1, self.n_states + 1, dtype=jnp.float64)[
            :, None
        ]  # Shape: (n_states, 1)
        j_indices = jnp.arange(1, self.n_controls + 1, dtype=jnp.float64)[
            None, :
        ]  # Shape: (1, n_controls)
        b_matrix = (i_indices - j_indices) * inv_nx_ny

        # C(i,j) = mu * (i+j) / (NX+NY) for i=1..NY, j=1..NX
        c_matrix = (i_indices + j_indices) * mu_over_nx_ny

        return b_matrix, c_matrix

    def objective(self, y, args):
        # Extract control and state variables
        # X(t,i) for t=1..N-1, i=1..NX
        # Y(t,i) for t=1..N, i=1..NY (Y(1,i) are fixed at 0 but included in y)
        x_vars = y[: (self.n_periods - 1) * self.n_controls].reshape(
            self.n_periods - 1, self.n_controls
        )
        y_vars = y[(self.n_periods - 1) * self.n_controls :].reshape(
            self.n_periods, self.n_states
        )

        # Vectorized objective computation
        # Objective terms for control variables: sum (X(t,i) + 0.5)^4
        x_terms = (x_vars + 0.5) ** 4
        x_objective = jnp.sum(x_terms)

        # Objective terms for state variables: sum (Y(t,i) + 0.25)^4
        y_terms = (y_vars + 0.25) ** 4
        y_objective = jnp.sum(y_terms)

        return x_objective + y_objective

    def constraint(self, y):
        b_matrix, c_matrix = self.args

        # Extract control and state variables
        # X(t,i) for t=1..N-1, i=1..NX
        # Y(t,i) for t=1..N, i=1..NY (Y(1,i) are fixed at 0 but included in y)
        x_vars = y[: (self.n_periods - 1) * self.n_controls].reshape(
            self.n_periods - 1, self.n_controls
        )
        y_vars = y[(self.n_periods - 1) * self.n_controls :].reshape(
            self.n_periods, self.n_states
        )

        # Vectorized constraint computation
        # Shape: (n_periods-1, n_states)
        y_current = y_vars[:-1]  # Y(t) for t=0..N-2
        y_next = y_vars[1:]  # Y(t+1) for t=0..N-2

        # Control contribution: B @ X(t) for each time period
        # Shape: (n_periods-1, n_states)
        control_contrib = x_vars @ b_matrix.T

        # Nonlinear element contribution: sum of C(i,j) * X(t,j) * Y(t,i)
        # According to the SIF GROUP USES, ALL elements E(T,K) are added to EACH
        # constraint. This means each constraint gets the same total:
        # sum over all i,j of C(i,j)*X(t,j)*Y(t,i)
        # Shape: (n_periods-1,)
        nonlinear_sum = jnp.einsum("ij,tj,ti->t", c_matrix, x_vars, y_current)
        # Broadcast to all states
        # Shape: (n_periods-1, n_states)
        nonlinear_contrib = nonlinear_sum[:, None]

        # Base transition terms: -Y(t+1) + 0.5*Y(t)
        base_terms = -y_next + 0.5 * y_current

        # State coupling terms
        # First state (j=0): +0.25*Y(t,1)
        state_coupling = jnp.zeros_like(y_current)
        state_coupling = state_coupling.at[:, 0].add(0.25 * y_current[:, 1])

        # Middle states (j=1..n_states-2): -0.25*Y(t,j-1) + 0.25*Y(t,j+1)
        if self.n_states > 2:
            state_coupling = state_coupling.at[:, 1:-1].add(
                -0.25 * y_current[:, :-2] + 0.25 * y_current[:, 2:]
            )

        # Last state (j=n_states-1): -0.25*Y(t,n_states-2)
        state_coupling = state_coupling.at[:, -1].add(-0.25 * y_current[:, -2])

        # Combine all terms
        equality_constraints = (
            base_terms + state_coupling + control_contrib + nonlinear_contrib
        )

        # Flatten to match expected output shape
        equality_constraints = equality_constraints.flatten()

        return equality_constraints, None

    @property
    def bounds(self):
        # Y(1,i) variables are fixed at 0, others are free
        n_x = (self.n_periods - 1) * self.n_controls

        # Lower bounds
        lb = jnp.full(self.n, -jnp.inf)
        # Fix Y(1,i) to 0
        lb = lb.at[n_x : n_x + self.n_states].set(0.0)

        # Upper bounds
        ub = jnp.full(self.n, jnp.inf)
        # Fix Y(1,i) to 0
        ub = ub.at[n_x : n_x + self.n_states].set(0.0)

        return lb, ub

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file: S(1000,2,4) 7.13884991815
        return jnp.array(7.13884991815)
