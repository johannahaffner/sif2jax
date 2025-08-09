# GPS Pattern Examples: From SIF to Vectorized JAX

## Example 1: ARWHEAD - Star Structure GPS

### SIF Structure
```sif
# Elements: interactions between x[i] and x[n]
ELEMENT TYPE EV ARROW
 VARIABLES V1 VN
 FORMULA (V1^2 + VN^2)^2

# Groups: sum over all elements
DO I 1 N-1
 XE ELEM(I) X(I) 1.0
 XE ELEM(I) X(N) 1.0
ND
```

### Vectorized JAX Implementation
```python
class ARWHEAD(AbstractUnconstrainedMinimisation):
    def __init__(self, n: int = 1000):
        self.n = n
        # Pre-compute indices for vectorization
        self.head_indices = jnp.arange(n-1)
        self.tail_index = n-1
        
    def objective(self, y: ArrayLike) -> Scalar:
        # Vectorized element computation
        # All elements computed simultaneously
        head_vars = y[self.head_indices]  # Shape: (n-1,)
        tail_var = y[self.tail_index]      # Scalar
        
        # Element function: (v1^2 + vn^2)^2
        element_values = (head_vars**2 + tail_var**2)**2
        
        # GPS aggregation (sum of elements)
        group_sum = jnp.sum(element_values)
        
        # Additional linear terms from SIF
        linear_term = -4 * jnp.sum(head_vars)
        
        return group_sum + linear_term + 3*(self.n-1)
```

## Example 2: CHAINWOO - Chain Structure GPS

### SIF Structure  
```sif
# Pairwise interactions along a chain
ELEMENT TYPE EV LINK
 VARIABLES X Y
 FORMULA 100*(Y - X^2)^2 + (1 - X)^2

DO I 1 N-1
 XE CHAIN(I) X(I) 1.0
 XE CHAIN(I) X(I+1) 1.0
ND
```

### Vectorized JAX Implementation
```python
class CHAINWOO(AbstractUnconstrainedMinimisation):
    def __init__(self, n: int = 100):
        self.n = n
        # Pre-compute for chain structure
        self.first_indices = jnp.arange(n-1)
        self.second_indices = jnp.arange(1, n)
        
    def objective(self, y: ArrayLike) -> Scalar:
        # Extract paired variables (vectorized)
        x_vals = y[self.first_indices]   # x[0] to x[n-2]
        y_vals = y[self.second_indices]  # x[1] to x[n-1]
        
        # Vectorized element evaluation
        # All chain links computed in parallel
        rosenbrock_terms = 100 * (y_vals - x_vals**2)**2
        linear_terms = (1 - x_vals)**2
        
        # GPS sum
        return jnp.sum(rosenbrock_terms + linear_terms)
```

## Example 3: DIXMAANJ - Sparse Quadratic GPS

### SIF Structure
```sif
# Multiple element types with different sparsity patterns
ELEMENT TYPE EV SQR
 VARIABLES V
 FORMULA ALPHA * V^2

ELEMENT TYPE EV SQRDIF  
 VARIABLES V1 V2
 FORMULA BETA * (V1 - V2)^2

ELEMENT TYPE EV PROD4
 VARIABLES V1 V2 V3 V4
 FORMULA GAMMA * V1 * V2 * V3 * V4
```

### Vectorized JAX Implementation
```python
class DIXMAANJ(AbstractUnconstrainedMinimisation):
    def __init__(self, n: int = 99, alpha=1.0, beta=0.0625, gamma=0.0625, delta=0.0625):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # Pre-compute indices for each element type
        # Type 1: Single variable squares
        self.sqr_indices = jnp.arange(n)
        
        # Type 2: Adjacent differences
        self.diff_first = jnp.arange(n-1)
        self.diff_second = jnp.arange(1, n)
        
        # Type 3: Sparse products (every 4th variable)
        m = n // 4
        self.prod_indices = jnp.arange(m) * 4
        
    def objective(self, y: ArrayLike) -> Scalar:
        # Element Type 1: Squares (all variables)
        sqr_elements = self.alpha * y**2
        
        # Element Type 2: Squared differences (vectorized)
        diff_elements = self.beta * (y[self.diff_first] - y[self.diff_second])**2
        
        # Element Type 3: Products (vectorized over sparse subset)
        prod_vars = y[self.prod_indices]
        prod_elements = self.gamma * prod_vars**4  # Simplified from 4-way product
        
        # GPS aggregation: sum all element groups
        return jnp.sum(sqr_elements) + jnp.sum(diff_elements) + jnp.sum(prod_elements)
```

## Example 4: ENGVAL1 - Complex GPS with Multiple Groups

### SIF Structure
```sif
# Different group types for objective and constraints
GROUPS
 N  OBJECTIVE
 G  CONSTR1
 L  CONSTR2

ELEMENT TYPE EV SQUARE
ELEMENT TYPE EV PRODUCT
ELEMENT TYPE EV RATIO

# Complex element-to-group mapping
```

### Vectorized JAX Implementation
```python
class ENGVAL1(AbstractConstrainedMinimisation):
    def __init__(self, n: int = 50):
        self.n = n
        
        # Pre-compute element-to-group mappings
        # Use sparse matrices for efficiency
        n_elements = n * (n-1) // 2
        n_groups = 3  # objective + 2 constraints
        
        # Element indices (upper triangular)
        self.elem_i, self.elem_j = jnp.triu_indices(n, k=1)
        
        # Group membership matrix (sparse)
        # Row i = element i, Col j = group j, Value = weight
        self.group_weights = self._build_group_matrix()
        
    def _build_group_matrix(self):
        """Build sparse element-to-group weight matrix"""
        # This would be pre-computed from SIF structure
        weights = jnp.ones((len(self.elem_i), 3))
        weights = weights.at[:, 0].set(1.0)  # Objective weights
        weights = weights.at[:, 1].set(0.5)  # Constraint 1 weights  
        weights = weights.at[:, 2].set(0.25) # Constraint 2 weights
        return weights
        
    def objective(self, y: ArrayLike) -> Scalar:
        # Compute all elements in parallel
        elements = self._compute_all_elements(y)
        
        # Extract objective group (column 0)
        obj_elements = elements * self.group_weights[:, 0]
        return jnp.sum(obj_elements)
        
    def constraints(self, y: ArrayLike) -> Array:
        # Compute all elements once
        elements = self._compute_all_elements(y)
        
        # Vectorized constraint aggregation
        # Matrix multiply: elements @ group_weights gives all constraints
        constraint_values = elements @ self.group_weights[:, 1:]
        return constraint_values
        
    def _compute_all_elements(self, y: ArrayLike) -> Array:
        """Compute all element values in parallel"""
        # Extract variable pairs
        vi = y[self.elem_i]
        vj = y[self.elem_j]
        
        # Different element types (vectorized)
        # Could use jnp.where for conditional element types
        return (vi - vj)**2  # Simplified example
```

## Example 5: PENALTY1 - Dense GPS with Regularization

### SIF Structure
```sif
# Sum of squares plus penalty term
ELEMENT TYPE EV RESSQR
 VARIABLES V
 FORMULA (V - TARGET)^2

ELEMENT TYPE EV PENALTY
 VARIABLES V
 FORMULA LAMBDA * V^4
```

### Vectorized JAX Implementation
```python
class PENALTY1(AbstractUnconstrainedMinimisation):
    def __init__(self, n: int = 100, lambda_reg: float = 1e-5):
        self.n = n
        self.lambda_reg = lambda_reg
        self.targets = jnp.arange(1, n+1) / n  # Pre-computed targets
        
    def objective(self, y: ArrayLike) -> Scalar:
        # Element Type 1: Residual squares (all computed at once)
        residual_elements = (y - self.targets)**2
        
        # Element Type 2: Penalty terms (vectorized)
        penalty_elements = self.lambda_reg * y**4
        
        # GPS aggregation
        residual_group = jnp.sum(residual_elements)
        penalty_group = jnp.sum(penalty_elements)
        
        return residual_group + penalty_group
```

## Example 6: Using JAX Segment Operations for GPS

### Advanced GPS Aggregation
```python
def gps_aggregate(element_values, element_to_group, group_weights, n_groups):
    """
    Efficient GPS aggregation using JAX segment operations
    
    Args:
        element_values: Array of computed element values
        element_to_group: Array mapping each element to its group
        group_weights: Weights for each element
        n_groups: Total number of groups
    """
    # Weight elements
    weighted = element_values * group_weights
    
    # Sum by group using segment_sum
    group_sums = jax.ops.segment_sum(
        weighted,
        element_to_group,
        num_segments=n_groups
    )
    
    return group_sums

# Alternative using scatter
def gps_aggregate_scatter(element_values, element_indices, group_indices, weights):
    """Using scatter for GPS aggregation when indices are irregular"""
    n_groups = group_indices.max() + 1
    result = jnp.zeros(n_groups)
    
    # Scatter-add weighted elements to groups
    result = result.at[group_indices].add(
        element_values[element_indices] * weights
    )
    
    return result
```

## Key Vectorization Principles

1. **Pre-compute all indices** in `__init__`
2. **Batch similar operations** using array operations
3. **Use segment operations** for group aggregation
4. **Avoid explicit loops** - use JAX primitives
5. **Share element computations** between objective and constraints
6. **Leverage sparsity** with index arrays rather than masks