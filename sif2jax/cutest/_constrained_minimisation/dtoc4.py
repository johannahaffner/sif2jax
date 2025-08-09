import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class DTOC4(AbstractConstrainedMinimisation):
    """A discrete time optimal control (DTOC) problem with quadratic objective.

    The system has N time periods, 1 control variable and 2 state variables.

    The problem is not convex.

    Sources: problem 4 in
    T.F. Coleman and A. Liao,
    "An Efficient Trust Region Method for Unconstrained Discrete-Time Optimal
    Control Problems",
    Tech. Report, ctc93tr144,  Advanced Computing Research Institute,
    Cornell University, 1992.

    G. Di Pillo, L. Grippo and F. Lampariello,
    "A class of structures quasi-Newton algorithms for optimal control
    problems",
    in H.E. Rauch, ed., IFAC Applications of nonlinear programming to
    optimization and control, pp. 101-107, IFAC, Pergamon Press, 1983.

    SIF input: Ph. Toint, August 1993

    classification QOR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters - using the default values from SIF
    n_periods: int = 1500  # N
    n_controls: int = 1  # NX=1
    n_states: int = 2  # NY=2

    # Total number of variables and constraints
    n: int = (n_periods - 1) * n_controls + n_periods * n_states  # 4499
    m: int = (n_periods - 1) * n_states  # 2998

    @property
    def y0(self):
        # Start from zeros except for fixed values
        y = jnp.zeros(self.n)
        # Y(1,1) = 0.0 is at index 1499
        # Y(1,2) = 1.0 is at index 1500
        y = y.at[1500].set(1.0)
        return y

    @property
    def args(self):
        # H = 1/N and related constants
        h = 1.0 / self.n_periods
        h5 = 5.0 * h
        return h, h5

    def objective(self, y, args):
        h, h5 = args
        # The SIF file says scale = "1/5H" but pycutest interprets this
        # as dividing by 5H, not multiplying by 1/(5H)
        scale_factor = h5  # 5H (reciprocal of what we had)

        # Extract control and state variables
        # Note: We don't enforce fixed variables here as pycutest evaluates
        # with the values given in the input vector
        # X(t) for t=1..N-1
        # Y(t,i) for t=1..N, i=1..2
        x_vars = y[: (self.n_periods - 1) * self.n_controls].reshape(
            self.n_periods - 1, self.n_controls
        )
        y_vars = y[(self.n_periods - 1) * self.n_controls :].reshape(
            self.n_periods, self.n_states
        )

        # Objective function has structure:
        # (1/5H) * [
        #   0.5*Y(1,1)^2 + 0.5*Y(1,2)^2 + X(1)^2 +
        #   sum_{t=2}^{N-1} (Y(t,1)^2 + Y(t,2)^2 + X(t)^2) +
        #   0.5*Y(N,1)^2 + 0.5*Y(N,2)^2
        # ]

        # First term (t=1)
        obj_sum = 0.5 * y_vars[0, 0] ** 2 + 0.5 * y_vars[0, 1] ** 2 + x_vars[0, 0] ** 2

        # Middle terms (t=2..N-1)
        if self.n_periods > 2:
            y_middle = y_vars[1:-1]  # Y(t) for t=2..N-1
            x_middle = x_vars[1:]  # X(t) for t=2..N-1
            obj_sum += jnp.sum(y_middle[:, 0] ** 2 + y_middle[:, 1] ** 2)
            obj_sum += jnp.sum(x_middle[:, 0] ** 2)

        # Last term (t=N)
        obj_sum += 0.5 * y_vars[-1, 0] ** 2 + 0.5 * y_vars[-1, 1] ** 2

        return scale_factor * obj_sum

    def constraint(self, y):
        h, h5 = self.args

        # Extract control and state variables
        # Note: We don't enforce fixed variables here as pycutest evaluates
        # with the values given in the input vector
        x_vars = y[: (self.n_periods - 1) * self.n_controls].reshape(
            self.n_periods - 1, self.n_controls
        )
        y_vars = y[(self.n_periods - 1) * self.n_controls :].reshape(
            self.n_periods, self.n_states
        )

        # Transition constraints
        # TT(t,1): -Y(t+1,1) + (1+5H)*Y(t,1) - 5H*Y(t,2) + 5H*X(t) -
        #          5H*Y(t,2)^2*Y(t,1) = 0
        # TT(t,2): -Y(t+1,2) + Y(t,2) + 5H*Y(t,1) = 0

        # Vectorize the computation
        y_current = y_vars[:-1]  # Y(t) for t=1..N-1
        y_next = y_vars[1:]  # Y(t+1) for t=1..N-1

        # Shape: (n_periods-1, n_states)
        constraints = jnp.zeros((self.n_periods - 1, self.n_states))

        # First constraint for each t: includes AAB element
        # TT(t,1) = -Y(t+1,1) + (1+5H)*Y(t,1) - 5H*Y(t,2) + 5H*X(t) - 5H*Y(t,2)^2*Y(t,1)
        aab_term = y_current[:, 1] ** 2 * y_current[:, 0]  # Y(t,2)^2 * Y(t,1)
        constraints = constraints.at[:, 0].set(
            -y_next[:, 0]
            + (1 + h5) * y_current[:, 0]
            - h5 * y_current[:, 1]
            + h5 * x_vars[:, 0]
            - h5 * aab_term
        )

        # Second constraint for each t
        # TT(t,2) = -Y(t+1,2) + Y(t,2) + 5H*Y(t,1)
        constraints = constraints.at[:, 1].set(
            -y_next[:, 1] + y_current[:, 1] + h5 * y_current[:, 0]
        )

        # Flatten to match expected output shape
        equality_constraints = constraints.flatten()

        return equality_constraints, None

    @property
    def bounds(self):
        # Y(1,1)=0, Y(1,2)=1 are fixed, others are free
        n_x = (self.n_periods - 1) * self.n_controls
        lbs = jnp.full(self.n, -jnp.inf)
        ubs = jnp.full(self.n, jnp.inf)

        # Y(1,1) at index n_x (1499) is fixed to 0.0
        lbs = lbs.at[n_x].set(0.0)
        ubs = ubs.at[n_x].set(0.0)

        # Y(1,2) at index n_x+1 (1500) is fixed to 1.0
        lbs = lbs.at[n_x + 1].set(1.0)
        ubs = ubs.at[n_x + 1].set(1.0)

        return lbs, ubs

    @property
    def expected_result(self) -> None:
        """Solution not provided in SIF file"""
        return None

    @property
    def expected_objective_value(self) -> None:
        """Optimal value for N=1500 from SIF file"""
        # Not provided for N=1500, but pattern suggests around 2.87
        return None
