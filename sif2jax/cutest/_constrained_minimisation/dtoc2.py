import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class DTOC2(AbstractConstrainedMinimisation):
    """A discrete time optimal control (DTOC) problem with trigonometric nonlinearities.

    The system has N time periods, NX control variables and NY state variables.

    The problem is not convex.

    Sources: problem 2 in
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

    classification OOR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters - using the default values from SIF
    n_periods: int = 1000  # N
    n_controls: int = 2  # NX
    n_states: int = 4  # NY

    # Total number of variables and constraints
    # Total variables include fixed Y(1,i) variables
    n: int = (n_periods - 1) * n_controls + n_periods * n_states  # 5998
    m: int = (n_periods - 1) * n_states  # 3996

    @property
    def y0(self):
        # Control variables start at 0.0
        x_vars = jnp.zeros((self.n_periods - 1) * self.n_controls)

        # State variables: Y(1,i) = i/(2*NY), others start at 0
        y_vars = jnp.zeros(self.n_periods * self.n_states)
        # Set initial Y(1,i) values
        for i in range(self.n_states):
            y_vars = y_vars.at[i].set((i + 1) / (2 * self.n_states))

        return jnp.concatenate([x_vars, y_vars])

    @property
    def args(self):
        # Build the transition matrix C
        two_ny = 2 * self.n_states
        inv_two_ny = 1.0 / two_ny

        # C(i,j) = (i+j) / (2*NY) for i=1..NY, j=1..NX
        i_indices = jnp.arange(1, self.n_states + 1, dtype=jnp.float64)[
            :, None
        ]  # Shape: (n_states, 1)
        j_indices = jnp.arange(1, self.n_controls + 1, dtype=jnp.float64)[
            None, :
        ]  # Shape: (1, n_controls)
        c_matrix = (i_indices + j_indices) * inv_two_ny

        return c_matrix

    def objective(self, y, args):
        # Extract control and state variables
        # X(t,i) for t=1..N-1, i=1..NX
        # Y(t,i) for t=1..N, i=1..NY
        x_vars = y[: (self.n_periods - 1) * self.n_controls].reshape(
            self.n_periods - 1, self.n_controls
        )
        y_vars = y[(self.n_periods - 1) * self.n_controls :].reshape(
            self.n_periods, self.n_states
        )

        # Objective for t=1..N-1
        # For each t: Y(t)^T * Y(t) * (sin^2(0.5 * ||X(t)||^2) + 1)
        # Vectorize the computation
        x_norm_sq = jnp.sum(x_vars**2, axis=1)  # ||X(t)||^2 for each t
        y_norm_sq = jnp.sum(y_vars[:-1] ** 2, axis=1)  # ||Y(t)||^2 for t=0..N-2
        sin_terms = jnp.sin(0.5 * x_norm_sq)
        obj_sum = jnp.sum(y_norm_sq * (sin_terms**2 + 1.0))

        # Add final term: Y(N)^T * Y(N)
        obj_sum += jnp.sum(y_vars[-1] ** 2)

        return obj_sum

    def constraint(self, y):
        c_matrix = self.args

        # Extract control and state variables
        # X(t,i) for t=1..N-1, i=1..NX
        # Y(t,i) for t=1..N, i=1..NY
        x_vars = y[: (self.n_periods - 1) * self.n_controls].reshape(
            self.n_periods - 1, self.n_controls
        )
        y_vars = y[(self.n_periods - 1) * self.n_controls :].reshape(
            self.n_periods, self.n_states
        )

        # Transition constraints: -Y(t+1,j) + sin(Y(t,j)) + sum_i C(j,i)*sin(X(t,i)) = 0
        # Vectorize the computation

        # Shape: (n_periods-1, n_states)
        y_current = y_vars[:-1]  # Y(t) for t=0..N-2
        y_next = y_vars[1:]  # Y(t+1) for t=0..N-2

        # Base terms: -Y(t+1,j) + sin(Y(t,j))
        base_constraints = -y_next + jnp.sin(y_current)

        # Control contribution: sum_i C(j,i)*sin(X(t,i))
        # Shape: (n_periods-1, n_controls)
        sin_x = jnp.sin(x_vars)
        # Shape: (n_periods-1, n_states)
        control_contrib = sin_x @ c_matrix.T

        # Combine all terms
        equality_constraints = base_constraints + control_contrib

        # Flatten to match expected output shape
        equality_constraints = equality_constraints.flatten()

        return equality_constraints, None

    @property
    def bounds(self):
        # Y(1,i) variables are fixed at i/(2*NY), others are free
        n_x = (self.n_periods - 1) * self.n_controls

        # Lower bounds
        lb = jnp.full(self.n, -jnp.inf)
        # Fix Y(1,i) to i/(2*NY)
        for i in range(self.n_states):
            lb = lb.at[n_x + i].set((i + 1) / (2 * self.n_states))

        # Upper bounds
        ub = jnp.full(self.n, jnp.inf)
        # Fix Y(1,i) to i/(2*NY)
        for i in range(self.n_states):
            ub = ub.at[n_x + i].set((i + 1) / (2 * self.n_states))

        return lb, ub

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file: SOLUTION(1000) 0.490200910983
        return jnp.array(0.490200910983)
