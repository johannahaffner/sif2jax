import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed - objective function discrepancy
# Attempts made:
# - Verified formula matches SIF file structure exactly
# - Correctly implemented group-separable structure with L2 groups
# - Applied scaling as per SIF documentation (divide by scale after group function)
# - Verified dimensions and starting values match pycutest
# Current status:
# - Our objective: 8,602,779.1 vs pycutest: 296,931,369.1 (factor of ~34.5)
# - Implementation follows SIF file correctly but differs from pycutest
# Suspected issues:
# - Possible undocumented transformation in pycutest's SIF interpreter
# - May need to revisit during optimization testing to understand discrepancy
# Additional resources needed:
# - Deep dive into pycutest's Fortran implementation
# - Comparison with optimization trajectories
class LUKVLE2(AbstractConstrainedMinimisation):
    """LUKVLE2 - Chained Wood function with Broyden banded constraints.

    Problem 5.2 from Luksan and Vlcek test problems.

    The objective is a modified chained Wood function:
    f(x) = Σ[i=1 to n/2-1] [
        0.01 * (-x_{2i} + x_{2i-1}²)² +
        (x_{2i-1} - 1)² +
        (1/90) * (-x_{2i+2} + x_{2i+1}²)² +
        (-x_{2i+1} - 1)² +
        0.1 * (x_{2i} + x_{2i+2} - 2)² +
        10.0 * (x_{2i} - x_{2i-1})²
    ]

    Subject to equality constraints:
    c_k(x) = 2x_k + 5x_k^3 + Σ[i=k-5 to k+1] (x_i + x_i^2) - 1 = 0,
    for k = 6, ..., n-2

    Starting point: x_i = -2 for i odd, x_i = 1 for i even

    Source: L. Luksan and J. Vlcek,
    "Sparse and partially separable test problems for
    unconstrained and equality constrained optimization",
    Technical Report 767, Inst. Computer Science, Academy of Sciences
    of the Czech Republic, 182 07 Prague, Czech Republic, 1999

    SIF input: Nick Gould, April 2001

    Classification: OOR2-AY-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 10000  # Default dimension, can be overridden
    # TODO set minimum dimension

    def objective(self, y, args):
        del args
        n = y.size

        a = y[0 : n - 3 : 2]  # 2i - 1  |  max: n - 3
        b = y[1 : n - 2 : 2]  # 2i      |  max: n - 2
        c = y[2 : n - 1 : 2]  # 2i + 1  |  max: n - 1
        d = y[3:n:2]  # 2i + 2  |  max: n

        e = 100 * ((a**2 - b) ** 2)  # Note: SIF file differs from source
        f = (a - 1) ** 2
        g = 90 * ((c**2 - d) ** 2)
        h = (c + 1) ** 2
        i = 10 * ((b + d - 2) ** 2)
        j = 0.1 * ((b - a) ** 2)

        return jnp.sum(e + f + g + h + i + j)

    def y0(self):
        # Starting point: x_i = -2 for i odd, x_i = 1 for i even
        y = jnp.zeros(self.n)
        # JAX uses 0-based indexing, so odd indices in the problem are even in JAX
        y = y.at[::2].set(-2.0)  # i = 1, 3, 5, ... (1-based) -> 0, 2, 4, ... (0-based)
        y = y.at[1::2].set(1.0)  # i = 2, 4, 6, ... (1-based) -> 1, 3, 5, ... (0-based)
        return y

    def args(self):
        return None

    def expected_result(self):
        # Solution is all ones
        return jnp.ones(self.n)

    def expected_objective_value(self):
        return jnp.array(0.0)

    def bounds(self):
        return None

    def constraint(self, y):
        n = len(y)
        if n < 8:  # Need at least 8 elements for constraints to start at k=6
            return jnp.array([]), None

        # Constraints from k=6 to n-2 (1-based) -> k=5 to n-3 (0-based)
        num_constraints = n - 7  # n-3 - 5 + 1
        if num_constraints <= 0:
            return jnp.array([]), None

        k_indices = jnp.arange(5, n - 2)

        # For each k, compute 2x_k + 5x_k^3 - 1
        x_k = y[k_indices]
        main_terms = 2 * x_k + 5 * x_k**3 - 1

        # For the sum part, we need to handle variable window sizes
        # This is trickier to vectorize perfectly, but we can use a sliding window
        # For now, keep a hybrid approach for the sum part
        sum_terms = jnp.zeros(num_constraints)

        for idx, k in enumerate(k_indices):
            start_idx = max(0, k - 5)
            end_idx = min(n - 1, k + 1)
            window = y[start_idx : end_idx + 1]
            sum_terms = sum_terms.at[idx].set(jnp.sum(window + window**2))

        constraints = main_terms + sum_terms
        return constraints, None
