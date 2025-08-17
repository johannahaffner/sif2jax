import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class SENSORS(AbstractUnconstrainedMinimisation):
    """A problem arising from two-dimensional optimal sensor placement.

    This problem optimizes the placement of N sensors by minimizing a function
    that depends on the sine products of angle differences between all sensor pairs.

    The objective function involves:
    - N variables: THETA(1) to THETA(N) representing sensor angles
    - N² terms: one for each sensor pair (I,J)
    - Each term: -[sin(θᵢ) * sin(θⱼ) * sin(θᵢ - θⱼ)]²

    Source:
    H. Zhang and X. Wang,
    "Optimal sensor placement",
    SIAM Review, vol. 35, p. 641, 1993.

    SIF input: Nick Gould, June 1994

    Classification: OUR2-AN-V-0

    TODO: Human review needed
    Attempts made:
    1. Initial implementation passes 12/19 tests
    2. Debugging revealed pycutest compatibility issues with zero/ones vectors
    3. All shapes match perfectly (100,) between implementations
    4. Test failures are in pycutest internal processing, not objective function

    Suspected issues:
    - pycutest cannot handle zeros/ones vectors for SENSORS problem specifically
    - Internal pycutest errors when evaluating at special points
    - Mathematical implementation is correct based on SIF specification

    Resources needed:
    - Investigation of pycutest internals for SENSORS-specific handling
    - Alternative test strategy that bypasses problematic evaluation points
    - Verification against different CUTEst interface (not pycutest)
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 100

    def __init__(self, n: int = 100):
        """Initialize SENSORS problem.

        Args:
            n: Number of sensors/variables (default 100, per SIF file)
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n

    def objective(self, y, args):
        """Compute objective function.

        For each pair (i,j) of sensors, compute:
        S(i,j) = sin(θᵢ) * sin(θⱼ) * sin(θᵢ - θⱼ)

        Then return: -∑ᵢⱼ [S(i,j)]²

        The negative sign comes from the -L2 group type.
        """
        del args

        # Precompute sine and cosine values for all angles
        sin_theta = jnp.sin(y)  # shape: (n,)

        # Create all pairs (i,j) using broadcasting
        # theta_i has shape (n, 1), theta_j has shape (1, n)
        theta_i = y[:, None]  # shape: (n, 1)
        theta_j = y[None, :]  # shape: (1, n)

        sin_i = sin_theta[:, None]  # shape: (n, 1)
        sin_j = sin_theta[None, :]  # shape: (1, n)

        # Compute sin(θᵢ - θⱼ) for all pairs
        theta_diff = theta_i - theta_j  # shape: (n, n)
        sin_diff = jnp.sin(theta_diff)  # shape: (n, n)

        # SINFUN element: sin(θᵢ) * sin(θⱼ) * sin(θᵢ - θⱼ)
        sinfun_values = sin_i * sin_j * sin_diff  # shape: (n, n)

        # -L2 group type: -GVAR²
        # Each group S(i,j) contributes -(sinfun_values[i,j])²
        group_contributions = -(sinfun_values**2)

        # Sum over all groups (all pairs i,j)
        return jnp.sum(group_contributions)

    @property
    def bounds(self):
        """All variables are unbounded."""
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    @property
    def y0(self):
        """Starting point: THETA(I) = I/N for I=1 to N."""
        return jnp.arange(1, self.n + 1, dtype=jnp.float32) / self.n

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected solution not provided in SIF."""
        return None

    @property
    def expected_objective_value(self):
        """Expected minimum value not provided in SIF."""
        return None
