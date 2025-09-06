import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed
# Attempts made: [multiple constraint formulation attempts, vectorized implementation]
# Suspected issues: [complex SIF constraint structure interpretation,
#                    constraint value discrepancies of ~2.0]
# Resources needed: [detailed SIF constraint pattern analysis,
#                    comparison with similar problems]


class MADSSCHJ(AbstractConstrainedMinimisation):
    """MADSSCHJ - A nonlinear minmax problem with variable dimension.

    N = 200 (number of variables - 1), n = 201 variables, m = 398 constraints.
    The Jacobian of the constraints is dense.

    Minimize Z subject to:
    Complex pattern of constraints involving linear combinations with quadratic terms.

    Start: X(i) = 10.0 for i = 1,...,N, Z = 0.0.

    Source: K. Madsen and H. Schjaer-Jacobsen,
    "Linearly Constrained Minmax Optimization",
    Mathematical Programming 14, pp. 208-223, 1978.

    SIF input: Ph. Toint, August 1993.

    Classification: LQR2-AN-V-V (parameterized, using N=200)
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Class constants
    N: int = 200  # Number of X variables
    M: int = 398  # Number of constraints = 2*N - 2

    def objective(self, y, args):
        del args
        # Objective: minimize Z (last variable)
        return y[-1]

    @property
    def y0(self):
        # Starting point: X(i) = 10.0 for i = 1,...,N, Z = 0.0
        x_start = jnp.full(MADSSCHJ.N, 10.0)
        z_start = jnp.array([0.0])
        return jnp.concatenate([x_start, z_start])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        # From SIF file: solution for N=200 is approximately -4992.1339031
        return jnp.array(-4992.1339031)

    @property
    def bounds(self):
        return None

    def constraint(self, y):
        """
        Build constraint vector for MADSSCHJ problem.
        From SIF file analysis:
        - All constraints are in XG format (≥ 0 after conversion)
        - Constants section adds -1.0 to all constraints
        - GROUP USES shows specific quadratic terms
        """
        x = y[:-1]  # X(1), ..., X(N)
        z = y[-1]  # Z

        constraints = []

        # First constraint C1: Z - sum(X[i] for i=2..N) - X[1]^2 - 1.0 ≥ 0
        c1 = z - jnp.sum(x[1:]) - x[0] ** 2 - 1.0
        constraints.append(c1)

        # Second constraint C2: Z - X[1] - sum(X[i] for i=3..N) - X[2]^2 - 1.0 ≥ 0
        c2 = z - x[0] - jnp.sum(x[2:]) - x[1] ** 2 - 1.0
        constraints.append(c2)

        # Third constraint C3: Z - X[1] - sum(X[i] for i=3..N) - 2*X[2]^2 - 1.0 ≥ 0
        c3 = z - x[0] - jnp.sum(x[2:]) - 2 * x[1] ** 2 - 1.0
        constraints.append(c3)

        # Intermediate constraints (blocks of 2): K from 4 to M-1, step 2
        k_values = jnp.arange(4, MADSSCHJ.M, 2)  # K = 4, 6, 8, ..., M-2

        def build_constraint_pair(k):
            j = (k + 2) // 2  # J = (K+2)//2
            j_idx = j - 1  # Convert to 0-based index

            # Pattern from SIF:
            # C(K): Z - sum(X[i] for i=1..J-1) - sum(X[i] for i=J+1..N)
            #       - X[J]^2 - 1.0 ≥ 0
            # C(K+1): Z - sum(X[i] for i=1..J-1) - sum(X[i] for i=J+1..N)
            #         - 2*X[J]^2 - 1.0 ≥ 0

            # This is equivalent to: Z - (sum(X) - X[J-1]) - X[J-1]^2 - 1.0
            linear_sum = jnp.sum(x) - x[j_idx]

            ck = z - linear_sum - x[j_idx] ** 2 - 1.0
            ck1 = z - linear_sum - 2 * x[j_idx] ** 2 - 1.0

            return jnp.array([ck, ck1])

        # Vectorized computation for intermediate constraints
        if len(k_values) > 0:
            constraint_pairs = jnp.array([build_constraint_pair(k) for k in k_values])
            intermediate_constraints = constraint_pairs.reshape(-1)  # Flatten pairs
            constraints.extend(intermediate_constraints)

        # Last constraint C(M): Z - sum(X[i] for i=1..N-1) - X[N]^2 - 1.0 ≥ 0
        cm = z - jnp.sum(x[:-1]) - x[-1] ** 2 - 1.0
        constraints.append(cm)

        # Convert list to array and ensure correct shape
        inequality_constraints = jnp.array(constraints)

        return None, inequality_constraints
