import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._problem import AbstractConstrainedMinimisation


class ACOPP14(AbstractConstrainedMinimisation):
    """AC Optimal Power Flow (OPF) for IEEE 14 Bus Power Systems Test Case.

    The optimal power flow (OPF) problem seeks to minimize generation costs subject to
    meeting demand and to various physical and engineering constraints.
    This is a polar formulation of the AC optimal power flow problem.

    See:
    http://www.ee.washington.edu/research/pstca/

    Source:
    A.Castillo@jhu.edu

    SIF input: Nick Gould, August 2011.

    Classification: QOR2-AY-38-68
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # System parameters
    n_nodes: int = 14
    n_generators: int = 5
    n_lines: int = 20

    # Generator node mapping (1-indexed in SIF, converted to 0-indexed)
    gen_nodes: Array = eqx.field(init=False)

    # Quadratic cost coefficients for generators
    a_coeffs: Array = eqx.field(init=False)
    b_coeffs: Array = eqx.field(init=False)

    # Power demand at each node (MW and MVAr)
    pd: Array = eqx.field(init=False)
    qd: Array = eqx.field(init=False)

    # Admittance matrices
    G: Array = eqx.field(init=False)
    B: Array = eqx.field(init=False)

    # Line data
    line_limits: Array = eqx.field(init=False)
    line_from: Array = eqx.field(init=False)
    line_to: Array = eqx.field(init=False)

    # Variable bounds
    m_min: Array = eqx.field(init=False)
    m_max: Array = eqx.field(init=False)
    p_min: Array = eqx.field(init=False)
    p_max: Array = eqx.field(init=False)
    q_min: Array = eqx.field(init=False)
    q_max: Array = eqx.field(init=False)

    def __init__(self):
        """Initialize the AC OPF problem for 14-bus system."""
        # Generator node mapping (1-indexed in SIF, converted to 0-indexed)
        self.gen_nodes = jnp.array([0, 1, 2, 5, 7])  # nodes 1, 2, 3, 6, 8

        # Quadratic cost coefficients for generators
        self.a_coeffs = jnp.array([860.586, 5000.0, 200.0, 200.0, 200.0])
        self.b_coeffs = jnp.array([2000.0, 2000.0, 4000.0, 4000.0, 4000.0])

        # Power demand at each node (MW and MVAr)
        self.pd = jnp.array(
            [
                0.0,
                0.217,
                0.942,
                0.478,
                0.076,
                0.112,
                0.0,
                0.0,
                0.295,
                0.09,
                0.035,
                0.061,
                0.135,
                0.149,
            ]
        )
        self.qd = jnp.array(
            [
                0.0,
                0.127,
                0.19,
                -0.039,
                0.016,
                0.075,
                0.0,
                0.0,
                0.166,
                0.058,
                0.018,
                0.016,
                0.058,
                0.05,
            ]
        )

        # Admittance matrix (G + jB) - from SIF file data
        # Initialize conductance and susceptance matrices
        self._init_admittance_matrices()

        # Line thermal limits (MVA)
        self._init_line_limits()

        # Variable bounds
        self._init_bounds()

    def _init_admittance_matrices(self):
        """Initialize conductance (G) and susceptance (B) matrices."""
        # Initialize as zeros
        self.G = jnp.zeros((self.n_nodes, self.n_nodes))
        self.B = jnp.zeros((self.n_nodes, self.n_nodes))

        # Line data: (from_node, to_node, resistance, reactance, line_charging)
        # Note: converting from 1-indexed to 0-indexed
        line_data = [
            (0, 1, 0.01938, 0.05917, 0.0528),
            (0, 4, 0.05403, 0.22304, 0.0492),
            (1, 2, 0.04699, 0.19797, 0.0438),
            (1, 3, 0.05811, 0.17632, 0.034),
            (1, 4, 0.05695, 0.17388, 0.0346),
            (2, 3, 0.06701, 0.17103, 0.0128),
            (3, 4, 0.01335, 0.04211, 0.0),
            (3, 5, 0.0, 0.20912, 0.0),
            (3, 6, 0.0, 0.55618, 0.0),
            (4, 6, 0.0, 0.25202, 0.0),
            (5, 10, 0.09498, 0.1989, 0.0),
            (5, 11, 0.12291, 0.25581, 0.0),
            (5, 12, 0.06615, 0.13027, 0.0),
            (6, 7, 0.0, 0.17615, 0.0),
            (6, 8, 0.0, 0.11001, 0.0),
            (8, 9, 0.03181, 0.0845, 0.0),
            (8, 13, 0.12711, 0.27038, 0.0),
            (9, 10, 0.08205, 0.19207, 0.0),
            (11, 12, 0.22092, 0.19988, 0.0),
            (12, 13, 0.17093, 0.34802, 0.0),
        ]

        # Build admittance matrix from line data
        for i, j, r, x, b_line in line_data:
            if r == 0.0 and x == 0.0:
                continue  # Skip zero impedance lines

            # Series admittance
            z_squared = r * r + x * x
            g_series = r / z_squared if z_squared > 0 else 0.0
            b_series = -x / z_squared if z_squared > 0 else 0.0

            # Add to admittance matrix (symmetric)
            self.G = self.G.at[i, j].add(g_series)
            self.G = self.G.at[j, i].add(g_series)
            self.G = self.G.at[i, i].add(-g_series)
            self.G = self.G.at[j, j].add(-g_series)

            self.B = self.B.at[i, j].add(b_series)
            self.B = self.B.at[j, i].add(b_series)
            self.B = self.B.at[i, i].add(-b_series + b_line / 2)
            self.B = self.B.at[j, j].add(-b_series + b_line / 2)

    def _init_line_limits(self):
        """Initialize line thermal limits."""
        # Thermal limits for each line (MVA)
        self.line_limits = jnp.array(
            [
                4.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        )

        # Line endpoints (0-indexed)
        self.line_from = jnp.array(
            [0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 6, 8, 8, 9, 11, 12]
        )
        self.line_to = jnp.array(
            [1, 4, 2, 3, 4, 3, 4, 5, 6, 6, 10, 11, 12, 7, 8, 9, 13, 10, 12, 13]
        )

    def _init_bounds(self):
        """Initialize variable bounds."""
        # Voltage magnitude bounds
        self.m_min = jnp.full(self.n_nodes, 0.94)
        self.m_max = jnp.full(self.n_nodes, 1.06)

        # Real power generation bounds
        self.p_min = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.p_max = jnp.array([3.324, 1.4, 1.0, 1.0, 1.0])

        # Reactive power generation bounds
        self.q_min = jnp.array([-0.4, -0.4, 0.0, -0.06, -0.06])
        self.q_max = jnp.array([0.5, 0.5, 0.4, 0.24, 0.24])

    @property
    def n(self):
        """Total number of variables."""
        return 2 * self.n_nodes + 2 * self.n_generators

    @property
    def m(self):
        """Total number of constraints."""
        return 2 * self.n_nodes + 2 * self.n_lines

    def objective(self, y, args):
        """Compute objective value (generation cost)."""
        del args
        # Extract real power generation variables
        p = y[2 * self.n_nodes : 2 * self.n_nodes + self.n_generators]

        # Quadratic cost: sum_i (a_i * p_i^2 + b_i * p_i)
        return jnp.sum(self.a_coeffs * p**2 + self.b_coeffs * p)

    def constraint(self, y):
        """Compute both equality and inequality constraints."""
        # Extract variables
        angles = y[: self.n_nodes]
        magnitudes = y[self.n_nodes : 2 * self.n_nodes]
        p_gen = y[2 * self.n_nodes : 2 * self.n_nodes + self.n_generators]
        q_gen = y[2 * self.n_nodes + self.n_generators :]

        # Initialize generation arrays for all nodes
        p_all = jnp.zeros(self.n_nodes)
        q_all = jnp.zeros(self.n_nodes)
        p_all = p_all.at[self.gen_nodes].set(p_gen)
        q_all = q_all.at[self.gen_nodes].set(q_gen)

        # Compute power flow equations for equality constraints
        eq_constraints = []

        # Real power balance equations
        for i in range(self.n_nodes):
            p_flow = 0.0
            for j in range(self.n_nodes):
                angle_diff = angles[i] - angles[j]
                p_flow += magnitudes[j] * (
                    self.G[i, j] * jnp.cos(angle_diff)
                    + self.B[i, j] * jnp.sin(angle_diff)
                )
            p_flow *= magnitudes[i]
            # P_injection - P_demand = P_flow
            eq_constraints.append(p_flow - p_all[i] + self.pd[i])

        # Reactive power balance equations
        for i in range(self.n_nodes):
            q_flow = 0.0
            for j in range(self.n_nodes):
                angle_diff = angles[i] - angles[j]
                q_flow += magnitudes[j] * (
                    self.G[i, j] * jnp.sin(angle_diff)
                    - self.B[i, j] * jnp.cos(angle_diff)
                )
            q_flow *= magnitudes[i]
            # Q_injection - Q_demand = Q_flow
            eq_constraints.append(q_flow - q_all[i] + self.qd[i])

        equality_constraints = jnp.array(eq_constraints)

        # Compute inequality constraints (line flow limits)
        ineq_constraints = []

        # Line flow constraints (from both ends of each line)
        for line_idx in range(self.n_lines):
            i = self.line_from[line_idx]
            j = self.line_to[line_idx]

            # Power flow from node i to j
            angle_diff_ij = angles[i] - angles[j]

            # Real and reactive power flows from i to j
            g_ij = self.G[i, j]
            b_ij = self.B[i, j]

            # Get line charging from original data
            # This is a simplification - in full model would need shunt elements
            p_ij = (
                magnitudes[i]
                * magnitudes[j]
                * (g_ij * jnp.cos(angle_diff_ij) + b_ij * jnp.sin(angle_diff_ij))
                - magnitudes[i] ** 2 * g_ij
            )

            q_ij = (
                magnitudes[i]
                * magnitudes[j]
                * (g_ij * jnp.sin(angle_diff_ij) - b_ij * jnp.cos(angle_diff_ij))
                + magnitudes[i] ** 2 * b_ij
            )

            # Apparent power from i to j (should be <= limit)
            s_ij = jnp.sqrt(p_ij**2 + q_ij**2)
            # Convert to >= 0 form: limit - s_ij >= 0
            ineq_constraints.append(self.line_limits[line_idx] - s_ij)

            # Power flow from node j to i
            angle_diff_ji = -angle_diff_ij

            p_ji = (
                magnitudes[j]
                * magnitudes[i]
                * (g_ij * jnp.cos(angle_diff_ji) + b_ij * jnp.sin(angle_diff_ji))
                - magnitudes[j] ** 2 * g_ij
            )

            q_ji = (
                magnitudes[j]
                * magnitudes[i]
                * (g_ij * jnp.sin(angle_diff_ji) - b_ij * jnp.cos(angle_diff_ji))
                + magnitudes[j] ** 2 * b_ij
            )

            # Apparent power from j to i (should be <= limit)
            s_ji = jnp.sqrt(p_ji**2 + q_ji**2)
            # Convert to >= 0 form: limit - s_ji >= 0
            ineq_constraints.append(self.line_limits[line_idx] - s_ji)

        inequality_constraints = jnp.array(ineq_constraints)

        return equality_constraints, inequality_constraints

    def y0(self):
        """Initial guess for variables."""
        # From SIF file starting values
        angles = jnp.array(
            [
                0.0,
                -0.0869565,
                -0.222228,
                -0.179564,
                -0.153053,
                -0.248023,
                -0.281907,
                -0.280009,
                -0.263545,
                -0.267959,
                -0.269142,
                -0.259156,
                -0.263755,
                -0.274401,
            ]
        )

        magnitudes = jnp.array(
            [
                1.06,
                1.045,
                1.01,
                1.01767,
                1.01951,
                1.07,
                1.06152,
                1.09,
                1.05593,
                1.05099,
                1.05691,
                1.05519,
                1.05038,
                1.03594,
            ]
        )

        p_gen = jnp.array([2.324, 0.4, 0.0, 0.0, 0.0])
        q_gen = jnp.array([-0.169, 0.424, 0.234, 0.122, 0.174])

        return jnp.concatenate([angles, magnitudes, p_gen, q_gen])

    def bounds(self):
        """Get variable bounds."""
        # Angle bounds (effectively unbounded)
        angle_lower = jnp.full(self.n_nodes, -jnp.inf)
        angle_upper = jnp.full(self.n_nodes, jnp.inf)

        # Magnitude bounds
        mag_lower = self.m_min
        mag_upper = self.m_max

        # Power generation bounds
        p_lower = self.p_min
        p_upper = self.p_max

        q_lower = self.q_min
        q_upper = self.q_max

        # Concatenate all bounds
        lower = jnp.concatenate([angle_lower, mag_lower, p_lower, q_lower])
        upper = jnp.concatenate([angle_upper, mag_upper, p_upper, q_upper])

        return (lower, upper)

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value (not provided in SIF)."""
        return None
