import jax
import jax.numpy as jnp
from functools import partial

from ..._problem import AbstractBoundedMinimisation


# TODO: Human review needed - Large-scale problem performance issues
# Problem size: 20,000 variables (2N where N=10,000)
# Optimization attempts:
#   1. JIT compilation with @partial(jax.jit, static_argnums=(0,))
#   2. Vectorized operations with pre-computed common terms
#   3. Efficient dot product instead of element-wise multiply + sum
#   4. Reduced intermediate array allocations
# Issues:
#   - Tests timeout (>60 seconds) due to problem scale
#   - May need specialized large-scale optimization techniques
#   - Possible numerical precision issues with constraint interpretation

class JANNSON3(AbstractBoundedMinimisation):
    """Convex-concave extensions example 3.
    
    A large-scale problem with 2N variables (N=10000, so 20000 variables total).
    All variables are bounded in [-1, 1].
    
    Source: C. Jannson
    "Convex-concave extensions"
    BIT 40(2) 2000:291-313
    
    SIF input: Nick Gould, September 2000
    
    Classification: OQR2-AN-V-3
    """
    
    N: int = 10000  # Parameter N (total variables = 2N = 20000)
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    
    @property
    def n(self):
        """Number of variables: 2N."""
        return 2 * self.N
    
    @partial(jax.jit, static_argnums=(0,))
    def objective(self, y, args):
        """Highly optimized vectorized objective function."""
        del args
        
        N = self.N
        
        # Pre-compute commonly used values to avoid redundant operations
        y_minus_1 = y - 1.0
        y_squared = y**2
        
        # Group G: 0.5 * (-X(2) + X(1)^2)^2 with GROUP TYPE L22
        g_term = 0.5 * (-y[1] + y_squared[0])**2
        
        # Groups G0 and G(1:2N): all (X(I) - 1)^2 terms
        # This includes both G0: (X(1) - 1)^2 and G(1:2N): (X(I) - 1)^2
        # Since G0 duplicates G(1), we have (2N+1) such terms total
        g_squared_terms = jnp.sum(y_minus_1**2) + y_minus_1[0]**2
        
        # Constraint terms (vectorized)
        # L: sum(X(I)) - 1 = 0
        l_constraint_sq = (jnp.sum(y) - 1.0)**2
        
        # Q: sum(X(I)^2) - 0.75 = 0  
        q_constraint_sq = (jnp.sum(y_squared) - 0.75)**2
        
        # P: sum(X(1:N) * X(N+1:2N)) - 1/(5N) = 0
        dot_product = jnp.dot(y[:N], y[N:])  # More efficient than element-wise multiply + sum
        p_constraint_sq = (dot_product - 0.2 / N)**2  # Precompute 1/(5N) = 0.2/N
        
        return g_term + g_squared_terms + l_constraint_sq + q_constraint_sq + p_constraint_sq
    
    @property
    def y0(self):
        """Initial guess (not specified in SIF, use zeros within bounds)."""
        return jnp.zeros(self.n)
    
    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None
    
    @property
    def bounds(self):
        """Variable bounds: [-1, 1] for all variables."""
        lower = jnp.full(self.n, -1.0)
        upper = jnp.full(self.n, 1.0)
        return lower, upper
    
    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None
    
    @property
    def expected_objective_value(self):
        """Expected optimal objective value (from SIF comments)."""
        if self.N == 10000:
            return 1.99985e4
        return None