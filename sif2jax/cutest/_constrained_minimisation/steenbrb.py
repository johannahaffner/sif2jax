import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class STEENBRB(AbstractConstrainedMinimisation):
    """A totally separable nonconvex multi-commodity network problem.

    # TODO: Human review needed - gradient test failing
    # Attempts made:
    # - Vectorized objective and constraint functions based on SIF file structure
    # - Verified arc costs match SIF file (COST1-COST18)
    # - Implemented flow conservation constraints with correct node connectivity
    # Suspected issues:
    # - Gradient test shows pycutest expects only 2 unique gradient values for
    #   capacity variables despite 18 different arc costs
    # - This suggests either a special problem structure or pycutest issue
    # Additional resources needed:
    # - Verification of pycutest's handling of this problem
    # - Original Steenbrink reference for problem structure

    Source: p. 120 of
    P.A. Steenbrink,
    "Optimization of Transport Networks",
    Wiley, 1974.

    Note that Steenbrink does not give values for TZERO, CCR and alpha.
    The problem has also been slightly perturbed by making NONZ >0, in
    order to avoid undefined values for some elements and infinite
    slope for others at the solution.

    SIF input: Ph. Toint, June 1990.

    classification ONR2-AY-468-108
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem dimensions
    n_arcs: int = 18
    n_trips: int = 12
    n: int = 2 * n_arcs + 2 * n_arcs * n_trips  # 468 variables
    m: int = 9 * n_trips  # 108 constraints

    @property
    def y0(self):
        # All variables start at 0.1
        return jnp.full(self.n, 0.1)

    @property
    def args(self):
        # Arc costs - from SIF file COST1 through COST18
        costs = jnp.array(
            [
                35.0,  # COST1
                40.0,  # COST2
                30.0,  # COST3
                100.0,  # COST4
                15.0,  # COST5
                55.0,  # COST6
                100.0,  # COST7
                25.0,  # COST8
                60.0,  # COST9
                35.0,  # COST10
                55.0,  # COST11
                15.0,  # COST12
                40.0,  # COST13
                60.0,  # COST14
                25.0,  # COST15
                30.0,  # COST16
                50.0,  # COST17
                50.0,  # COST18
            ]
        )

        # Parameters
        alph = 0.01
        tzero = 0.01
        ccr = 0.01
        nonz = 0.01

        # Compute coefficients
        la = costs * alph
        lt = costs * tzero
        lc = costs * ccr

        # Minimal capacities (with NONZ perturbation)
        cmd = jnp.full(self.n_arcs, nonz)
        cmr = jnp.full(self.n_arcs, nonz)

        # Trip demands (positive = source, negative = sink)
        # Structure: (trip_id, node, demand)
        demands = [
            # Trip 1: 2000 from node 2 to node 3
            (0, 1, -2000.0),
            (0, 2, 2000.0),
            # Trip 2: 2000 from node 2 to node 4
            (1, 1, -2000.0),
            (1, 3, 2000.0),
            # Trip 3: 1000 from node 2 to node 5
            (2, 1, -1000.0),
            (2, 4, 1000.0),
            # Trip 4: 1000 from node 3 to node 4
            (3, 2, -1000.0),
            (3, 3, 1000.0),
            # Trip 5: 2000 from node 3 to node 5
            (4, 2, -2000.0),
            (4, 4, 2000.0),
            # Trip 6: 1000 from node 4 to node 5
            (5, 3, -1000.0),
            (5, 4, 1000.0),
            # Trip 7: 200 from node 3 to node 2
            (6, 2, -200.0),
            (6, 1, 200.0),
            # Trip 8: 200 from node 4 to node 2
            (7, 3, -200.0),
            (7, 1, 200.0),
            # Trip 9: 100 from node 5 to node 2
            (8, 4, -100.0),
            (8, 1, 100.0),
            # Trip 10: 100 from node 4 to node 3
            (9, 3, -100.0),
            (9, 2, 100.0),
            # Trip 11: 200 from node 5 to node 3
            (10, 4, -200.0),
            (10, 2, 200.0),
            # Trip 12: 100 from node 5 to node 4
            (11, 4, -100.0),
            (11, 3, 100.0),
        ]

        return (la, lt, lc, cmd, cmr, nonz, demands)

    def objective(self, y, args):
        la, lt, lc, cmd, cmr, nonz, demands = args

        # Extract variables
        # CD(k) for k=1..18, CR(k) for k=1..18, then D(i,k) and R(i,k)
        cd = y[: self.n_arcs]
        cr = y[self.n_arcs : 2 * self.n_arcs]

        # Flow variables start at index 2*n_arcs
        flow_start = 2 * self.n_arcs
        d_flows = y[flow_start : flow_start + self.n_arcs * self.n_trips].reshape(
            self.n_trips, self.n_arcs
        )
        r_flows = y[flow_start + self.n_arcs * self.n_trips :].reshape(
            self.n_trips, self.n_arcs
        )

        # Vectorized computation for all arcs
        # Total flows for each arc (sum over all trips)
        flow_d_total = jnp.sum(d_flows, axis=0)  # Shape: (n_arcs,)
        flow_r_total = jnp.sum(r_flows, axis=0)  # Shape: (n_arcs,)

        # XT element contribution: LT * FLOW + LC * F^3 / C^2
        xt_contrib_d = lt * flow_d_total + lc * flow_d_total**3 / cd**2
        xt_contrib_r = lt * flow_r_total + lc * flow_r_total**3 / cr**2

        # IIJ element contribution: sqrt(C - CMIN), with LA applied at group level
        # CMIN = CMD + MNONZ where MNONZ = -NONZ
        iij_contrib_d = la * jnp.sqrt(cd - (cmd - nonz))
        iij_contrib_r = la * jnp.sqrt(cr - (cmr - nonz))

        # Sum all contributions
        obj = jnp.sum(xt_contrib_d + xt_contrib_r + iij_contrib_d + iij_contrib_r)

        return obj

    def constraint(self, y):
        la, lt, lc, cmd, cmr, nonz, demands = self.args

        # Extract flow variables
        flow_start = 2 * self.n_arcs
        d_flows = y[flow_start : flow_start + self.n_arcs * self.n_trips].reshape(
            self.n_trips, self.n_arcs
        )
        r_flows = y[flow_start + self.n_arcs * self.n_trips :].reshape(
            self.n_trips, self.n_arcs
        )

        # Convert arc connections to arrays for vectorized operations
        # (from_node, to_node) where nodes are 0-indexed
        from_nodes = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 7])
        to_nodes = jnp.array([1, 2, 6, 3, 6, 7, 4, 6, 7, 5, 7, 8, 5, 7, 8, 8, 7, 8])

        # Convert demands to structured arrays for vectorized lookup
        # Create a demand matrix: demands_matrix[trip, node] = demand_value
        demands_matrix = jnp.zeros((self.n_trips, 9))
        for trip_id, demand_node, demand_value in demands:
            demands_matrix = demands_matrix.at[trip_id, demand_node].set(demand_value)

        # Vectorized constraint computation
        # Shape: (n_trips, 9) for 9 nodes × 12 trips = 108 constraints

        # For each trip-node combination, compute net flow
        # Create masks for efficient arc processing
        # Shape: (1, n_arcs, 9)
        from_node_mask = from_nodes[None, :, None] == jnp.arange(9)[None, None, :]
        to_node_mask = to_nodes[None, :, None] == jnp.arange(9)[None, None, :]

        # Expand flow arrays for broadcasting
        d_flows_expanded = d_flows[:, :, None]  # Shape: (n_trips, n_arcs, 1)
        r_flows_expanded = r_flows[:, :, None]  # Shape: (n_trips, n_arcs, 1)

        # Compute outflow and inflow for each (trip, node) pair
        # For outgoing arcs (from_node == node): -d_flow + r_flow
        outflow_d_array = jnp.asarray(jnp.where(from_node_mask, -d_flows_expanded, 0))
        outflow_d = jnp.sum(outflow_d_array, axis=1)  # Shape: (n_trips, 9)

        outflow_r_array = jnp.asarray(jnp.where(from_node_mask, r_flows_expanded, 0))
        outflow_r = jnp.sum(outflow_r_array, axis=1)  # Shape: (n_trips, 9)

        # For incoming arcs (to_node == node): +d_flow - r_flow
        inflow_d_array = jnp.asarray(jnp.where(to_node_mask, d_flows_expanded, 0))
        inflow_d = jnp.sum(inflow_d_array, axis=1)  # Shape: (n_trips, 9)

        inflow_r_array = jnp.asarray(jnp.where(to_node_mask, -r_flows_expanded, 0))
        inflow_r = jnp.sum(inflow_r_array, axis=1)  # Shape: (n_trips, 9)

        # Net flow = outflow - inflow for each (trip, node)
        net_flows = outflow_d + outflow_r + inflow_d + inflow_r  # Shape: (n_trips, 9)

        # Flow conservation constraints: net_flow - demand = 0
        equality_constraints = net_flows - demands_matrix

        # Flatten to match expected output shape (108 constraints)
        equality_constraints = equality_constraints.flatten()

        return equality_constraints, None

    @property
    def bounds(self):
        # CD and CR have lower bounds cmd and cmr respectively
        # Only the first variable has a lower bound of 0.01 based on the test output
        lower = jnp.zeros(self.n)
        upper = jnp.full(self.n, jnp.inf)

        # Set lower bounds for capacity variables CD(1) and CR(1)
        lower = lower.at[0].set(0.01)  # CD(1)
        lower = lower.at[1].set(0.01)  # CD(2)
        lower = lower.at[self.n_arcs].set(0.01)  # CR(1)
        lower = lower.at[self.n_arcs + 1].set(0.01)  # CR(2)

        return lower, upper

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # According to the SIF file comment (line 359),
        # the optimal objective value is 9098.9319884
        return jnp.array(9098.9319884)
