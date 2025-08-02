import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class DTOC3(AbstractConstrainedMinimisation):
    """A discrete time optimal control (DTOC) problem with quadratic objective.

    The system has N time periods, 1 control variable and 2 state variables.

    The problem is convex.

    Sources: problem 3 in
    T.F. Coleman and A. Liao,
    "An Efficient Trust Region Method for Unconstrained Discrete-Time Optimal
    Control Problems",
    Tech. Report, ctc93tr144,  Advanced Computing Research Institute,
    Cornell University, 1992.

    D.P. Bertsekas,
    "Projected Newton methods for optimization problems with simple
    constraints",
    SIAM J. Control and Optimization 20, pp. 221-246, 1982.

    SIF input: Ph. Toint, August 1993

    classification QLR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters - using the default values from SIF
    n_periods: int = 1500  # N
    n_controls: int = 1  # NX=1
    n_states: int = 2  # NY=2

    # Total number of variables and constraints
    # Total variables include fixed Y(1,i) variables
    n: int = (n_periods - 1) * n_controls + n_periods * n_states  # 4499
    m: int = (n_periods - 1) * n_states  # 2998

    @property
    def y0(self):
        # Control variables start at 0.0
        x_vars = jnp.zeros((self.n_periods - 1) * self.n_controls)

        # State variables: Y(1,1)=15, Y(1,2)=5, others start at 0
        y_vars = jnp.zeros(self.n_periods * self.n_states)
        y_vars = y_vars.at[0].set(15.0)  # Y(1,1)
        y_vars = y_vars.at[1].set(5.0)  # Y(1,2)

        return jnp.concatenate([x_vars, y_vars])

    @property
    def args(self):
        # Scaling factor S = 1/N
        s = 1.0 / self.n_periods
        return s

    def objective(self, y, args):
        s = args
        scale_factor = 2.0 / s

        # Extract control and state variables
        # X(t) for t=1..N-1
        # Y(t,i) for t=1..N, i=1..2
        x_vars = y[: (self.n_periods - 1) * self.n_controls].reshape(
            self.n_periods - 1, self.n_controls
        )
        y_vars = y[(self.n_periods - 1) * self.n_controls :].reshape(
            self.n_periods, self.n_states
        )

        # Objective: sum over t=1..N-1 of (2/S) * (2*Y(t+1,1)^2 + Y(t+1,2)^2 + 6*X(t)^2)
        # Vectorize the computation

        # Y(t+1) for t=0..N-2
        y_next = y_vars[1:]

        # Y(t+1,1)^2 and Y(t+1,2)^2
        y1_sq = y_next[:, 0] ** 2
        y2_sq = y_next[:, 1] ** 2

        # X(t)^2
        x_sq = x_vars[:, 0] ** 2

        obj_sum = scale_factor * jnp.sum(2.0 * y1_sq + y2_sq + 6.0 * x_sq)

        return obj_sum

    def constraint(self, y):
        s = self.args

        # Extract control and state variables
        # X(t) for t=1..N-1
        # Y(t,i) for t=1..N, i=1..2
        x_vars = y[: (self.n_periods - 1) * self.n_controls].reshape(
            self.n_periods - 1, self.n_controls
        )
        y_vars = y[(self.n_periods - 1) * self.n_controls :].reshape(
            self.n_periods, self.n_states
        )

        # Transition constraints
        # TT(t,1): -Y(t+1,1) + Y(t,1) + S*Y(t,2) = 0
        # TT(t,2): -Y(t+1,2) + Y(t,2) - S*Y(t,1) + S*X(t) = 0

        # Vectorize the computation
        # Shape: (n_periods-1, n_states)
        y_current = y_vars[:-1]  # Y(t) for t=0..N-2
        y_next = y_vars[1:]  # Y(t+1) for t=0..N-2

        # Shape: (n_periods-1, n_states)
        constraints = jnp.zeros((self.n_periods - 1, self.n_states))

        # First constraint for each t: -Y(t+1,1) + Y(t,1) + S*Y(t,2)
        constraints = constraints.at[:, 0].set(
            -y_next[:, 0] + y_current[:, 0] + s * y_current[:, 1]
        )

        # Second constraint for each t: -Y(t+1,2) + Y(t,2) - S*Y(t,1) + S*X(t)
        constraints = constraints.at[:, 1].set(
            -y_next[:, 1] + y_current[:, 1] - s * y_current[:, 0] + s * x_vars[:, 0]
        )

        # Flatten to match expected output shape
        equality_constraints = constraints.flatten()

        return equality_constraints, None

    @property
    def bounds(self):
        # Y(1,1)=15, Y(1,2)=5 are fixed, others are free
        n_x = (self.n_periods - 1) * self.n_controls

        # Lower bounds
        lb = jnp.full(self.n, -jnp.inf)
        # Fix Y(1,1) and Y(1,2)
        lb = lb.at[n_x].set(15.0)  # Y(1,1)
        lb = lb.at[n_x + 1].set(5.0)  # Y(1,2)

        # Upper bounds
        ub = jnp.full(self.n, jnp.inf)
        # Fix Y(1,1) and Y(1,2)
        ub = ub.at[n_x].set(15.0)  # Y(1,1)
        ub = ub.at[n_x + 1].set(5.0)  # Y(1,2)

        return lb, ub

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file: SOLUTION(1500) is not given, but we can interpolate
        # Between SOLUTION(1000)=235.182824435 and SOLUTION(5000)=235.154640099
        # For N=1500, a reasonable estimate would be around 235.17
        # However, let's use None since the exact value isn't provided
        return None
