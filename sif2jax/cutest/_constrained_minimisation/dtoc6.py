import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class DTOC6(AbstractConstrainedMinimisation):
    """A discrete time optimal control (DTOC) problem with nonlinear objective.

    The system has N time periods, 1 control variable and 1 state variable.

    The problem is convex.

    Sources: problem 6 in
    T.F. Coleman and A. Liao,
    "An Efficient Trust Region Method for Unconstrained Discrete-Time Optimal
    Control Problems",
    Tech. Report, ctc93tr144,  Advanced Computing Research Institute,
    Cornell University, 1992.

    D.M. Murray and S.J. Yakowitz,
    "The application of optimal control methodology to nonlinear programming
    problems",
    Mathematical Programming 21, pp. 331-347, 1981.

    SIF input: Ph. Toint, August 1993

    classification OOR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters - using the default values from SIF
    n_periods: int = 5001  # N
    n_controls: int = 1  # NX=1
    n_states: int = 1  # NY=1

    # Total number of variables and constraints
    n: int = (n_periods - 1) * n_controls + n_periods * n_states  # 10001
    m: int = n_periods - 1  # 5000

    @property
    def y0(self):
        # Start from zeros except for Y(1) = 0.0 (which is already 0)
        return jnp.zeros(self.n)

    @property
    def args(self):
        # No special args needed for this problem
        return None

    def objective(self, y, args):
        # Extract control and state variables
        # Note: We don't enforce fixed variables here as pycutest evaluates
        # with the values given in the input vector
        # X(t) for t=1..N-1
        # Y(t) for t=1..N
        x_vars = y[: self.n_periods - 1]
        y_vars = y[self.n_periods - 1 :]

        # Objective function: sum_{t=1}^{N-1} 2*(Y(t)^2 + exp(X(t))*Y(t) + X(t)^2)
        # Note: sum is over t=1..N-1, so we use Y(1)..Y(N-1)
        y_obj = y_vars[:-1]  # Y(t) for t=1..N-1

        # Compute exp(X(t))
        exp_x = jnp.exp(x_vars)

        # The objective groups OY(t) and OX(t) both use L2 group type with scale 2.0
        # L2 group type squares the GVAR
        # OY(t): 2.0 * (Y(t) + exp(X(t)))^2
        # OX(t): 2.0 * X(t)^2
        # Total objective: sum of OY(t) + OX(t)

        # OY(t) group: GVAR = Y(t), nonlinear element E(T) = exp(X(t))
        # L2 group: (GVAR + E(T))^2 * scale
        # The SIF file says scale = 2.0 but pycutest interprets this as dividing by 2
        oy_terms = 0.5 * (y_obj + exp_x) ** 2

        # OX(t) group: GVAR = X(t), no nonlinear element
        # L2 group: GVAR^2 * scale
        # The SIF file says scale = 2.0 but pycutest interprets this as dividing by 2
        ox_terms = 0.5 * x_vars**2

        obj_sum = jnp.sum(oy_terms + ox_terms)

        return obj_sum

    def constraint(self, y):
        # Extract control and state variables
        # Note: We don't enforce fixed variables here as pycutest evaluates
        # with the values given in the input vector
        x_vars = y[: self.n_periods - 1]
        y_vars = y[self.n_periods - 1 :]

        # Transition constraints with exponential term
        # TT(t): -Y(t+1) + Y(t) + exp(X(t)) = 0
        # for t=1..N-1

        y_current = y_vars[:-1]  # Y(t) for t=1..N-1
        y_next = y_vars[1:]  # Y(t+1) for t=1..N-1

        # Compute exp(X(t))
        exp_x = jnp.exp(x_vars)

        # Compute constraints
        constraints = -y_next + y_current + exp_x

        return constraints, None

    @property
    def bounds(self):
        # Y(1)=0.0 is fixed, others are free
        n_x = self.n_periods - 1

        lbs = jnp.full(self.n, -jnp.inf)
        ubs = jnp.full(self.n, jnp.inf)

        # Y(1) at index n_x is fixed to 0.0
        lbs = lbs.at[n_x].set(0.0)
        ubs = ubs.at[n_x].set(0.0)

        return lbs, ubs

    @property
    def expected_result(self) -> None:
        """Solution not provided in SIF file"""
        return None

    @property
    def expected_objective_value(self) -> None:
        """Optimal value for N=5001 not provided in SIF file"""
        return None
