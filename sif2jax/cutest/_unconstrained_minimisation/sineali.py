import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class SINEALI(AbstractUnconstrainedMinimisation):
    """
    TODO: Human review needed
    Attempts made:
    1. Initial implementation in unconstrained module but classification
       OBR2-AN-V-0 indicates bounded problem
    2. Problem has bounds: X1 ∈ [-5.66548, 1.5708], X(i) ∈ [-5.66548, 2.2214] for i ≥ 2
    3. Should be moved to bounded_minimisation module

    Suspected issues:
    - Wrong base class - should inherit from AbstractBoundedMinimisation
    - Implementation needs bounds handling
    - Gradient computation may need adjustment for bounded case

    Resources needed:
    - Move to bounded_minimisation module
    - Re-implement with proper bounds support
    """

    """A variation on the extended Rosenbrock function in which
    the squares are replaced by sines.
    
    This problem modifies the extended Rosenbrock function by replacing 
    squared terms with sine functions. The original squares in the 
    constraints are replaced by sines to avoid multiple minima.
    
    The objective function involves:
    - First term: sin(X1 - 1.0)
    - Remaining terms: sin(scale * (X(i) - X(i-1)²)) for i=2..N
    where scale = 0.01
    
    Source: an original idea by
    Ali Bouriacha, private communication.
    
    SIF input: Nick Gould and Ph. Toint, October, 1993.
    
    Classification: OBR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 1000
    scale: float = 0.01

    def __init__(self, n: int = 1000):
        self.n = n

    def objective(self, y, args):
        del args

        # First group: sin(X1 - 1.0)
        term1 = jnp.sin(y[0] - 1.0)

        # Remaining groups: sin(scale * (X(i) - X(i-1)²)) for i=2..N
        if self.n > 1:
            x_prev_squared = y[:-1] ** 2  # X(i-1)² for i=2..N
            differences = y[1:] - x_prev_squared  # X(i) - X(i-1)²
            scaled_diffs = self.scale * differences
            remaining_terms = jnp.sin(scaled_diffs)
            return term1 + jnp.sum(remaining_terms)
        else:
            return term1

    @property
    def y0(self):
        # Starting point: all zeros
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        # Expected solution: -100 * (N - 1) - 1
        return jnp.array(-100.0 * (self.n - 1) - 1.0)
