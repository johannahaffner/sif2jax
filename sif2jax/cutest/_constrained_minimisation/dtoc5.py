import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class DTOC5(AbstractConstrainedMinimisation):
    """A discrete time optimal control (DTOC) problem with quadratic constraints.

    The system has N time periods, 1 control variable and 1 state variable.

    The problem is convex.

    Sources: problem 5 in
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

    classification QQR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters - using the default values from SIF
    n_periods: int = 5000  # N
    n_controls: int = 1  # NX=1
    n_states: int = 1  # NY=1

    # Total number of variables and constraints
    n: int = (n_periods - 1) * n_controls + n_periods * n_states  # 9999
    m: int = n_periods - 1  # 4999

    @property
    def y0(self):
        # Start from zeros except for Y(1) = 1.0
        y = jnp.zeros(self.n)
        # Y(1) is at index n_periods-1 (after all X variables)
        y = y.at[self.n_periods - 1].set(1.0)
        return y

    @property
    def args(self):
        # H = 1/N
        h = 1.0 / self.n_periods
        return h

    def objective(self, y, args):
        # The SIF file says scale = "RN" but pycutest interprets this
        # as dividing by RN, not multiplying by RN
        scale_factor = 1.0 / float(self.n_periods)  # 1/RN

        # Extract control and state variables
        # Note: We don't enforce fixed variables here as pycutest evaluates
        # with the values given in the input vector
        # X(t) for t=1..N-1
        # Y(t) for t=1..N
        x_vars = y[: self.n_periods - 1]
        y_vars = y[self.n_periods - 1 :]

        # Objective function: N * sum_{t=1}^{N-1} (Y(t)^2 + X(t)^2)
        # Note: sum is over t=1..N-1, so we use Y(1)..Y(N-1)
        y_obj = y_vars[:-1]  # Y(t) for t=1..N-1
        obj_sum = jnp.sum(y_obj**2 + x_vars**2)

        return scale_factor * obj_sum

    def constraint(self, y):
        h = self.args

        # Extract control and state variables
        # Note: We don't enforce fixed variables here as pycutest evaluates
        # with the values given in the input vector
        x_vars = y[: self.n_periods - 1]
        y_vars = y[self.n_periods - 1 :]

        # Transition constraints with quadratic term
        # TT(t): -Y(t+1) + Y(t) - H*X(t) + H*Y(t)^2 = 0
        # for t=1..N-1

        y_current = y_vars[:-1]  # Y(t) for t=1..N-1
        y_next = y_vars[1:]  # Y(t+1) for t=1..N-1

        # Compute constraints
        constraints = -y_next + y_current - h * x_vars + h * y_current**2

        return constraints, None

    @property
    def bounds(self):
        # Y(1)=1.0 is fixed, others are free
        n_x = self.n_periods - 1

        lbs = jnp.full(self.n, -jnp.inf)
        ubs = jnp.full(self.n, jnp.inf)

        # Y(1) at index n_x is fixed to 1.0
        lbs = lbs.at[n_x].set(1.0)
        ubs = ubs.at[n_x].set(1.0)

        return lbs, ubs

    @property
    def expected_result(self) -> None:
        """Solution not provided in SIF file"""
        return None

    @property
    def expected_objective_value(self) -> None:
        """Optimal value for N=5000 from SIF file"""
        # return 1.531611890390  # Value for N=5000, but we use N=5001
        return None
