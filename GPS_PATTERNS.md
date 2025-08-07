# Group-Partially-Separable (GPS) Pattern Recognition Guide

## Understanding GPS Structure

The GPS structure is fundamental to most CUTEst problems. According to S2MPJ research, this structure enables efficient problem representation and computation.

## Core GPS Formula

```
f(x) = Σᵢ gᵢ(x) + ½xᵀQx + cᵀx
```

Where:
- `gᵢ(x)` are group functions (possibly nonlinear)
- `Q` is a quadratic term (often sparse/zero)
- `c` is a linear term

## GPS Pattern Categories

### 1. Sum of Squares Pattern
**SIF Indicators:**
- Multiple ELEMENT TYPE definitions
- Groups with "SQUARE" operations
- DO loops creating similar structures

**JAX Implementation:**
```python
def objective(self, y: ArrayLike) -> Scalar:
    # Compute residuals in batch
    residuals = self._compute_residuals(y)  # Shape: (n_residuals,)
    return 0.5 * jnp.sum(residuals ** 2)

def _compute_residuals(self, y: ArrayLike) -> Array:
    # Example: Model evaluation minus observations
    model_values = self._evaluate_model(y)
    return model_values - self.observations
```

### 2. Weighted Element Sum Pattern
**SIF Indicators:**
- GROUP definitions with scaling
- ELEMENT assignments to groups
- Weight parameters (GSCALE, ESCALE)

**JAX Implementation:**
```python
def objective(self, y: ArrayLike) -> Scalar:
    # Evaluate all elements
    element_values = vmap(self._eval_element)(
        self.element_params, y[self.element_var_indices]
    )
    
    # Apply group aggregation
    group_sums = jnp.zeros(self.n_groups)
    for g in range(self.n_groups):
        group_mask = self.element_to_group == g
        group_sums = group_sums.at[g].set(
            jnp.sum(element_values * group_mask * self.element_weights)
        )
    
    # Apply group scaling and sum
    return jnp.sum(group_sums * self.group_scales)
```

### 3. Constraint GPS Pattern
**SIF Indicators:**
- Multiple GROUP definitions of type 'E' or 'L'
- Constraints built from element combinations
- Range specifications

**JAX Implementation:**
```python
def constraints(self, y: ArrayLike) -> Array:
    # Compute constraint groups similarly to objective
    element_values = self._compute_constraint_elements(y)
    
    # Aggregate by constraint
    constraint_values = jnp.zeros(self.n_constraints)
    for c in range(self.n_constraints):
        mask = self.constraint_element_mask[c]
        constraint_values = constraint_values.at[c].set(
            jnp.sum(element_values * mask)
        )
    
    return constraint_values - self.constraint_rhs
```

## Identifying GPS in SIF Files

### Quick Identification Checklist

1. **Look for GROUPS section:**
```sif
GROUPS
 N  OBJECTIVE
 E  CONSTR1
 L  CONSTR2
```

2. **Check for ELEMENTS:**
```sif
ELEMENTS
 T  SQUARE
 V  X
 F  X * X
 T  PROD2
 V  X
 V  Y  
 F  X * Y
```

3. **Find element-to-group mappings:**
```sif
ELEMENT USES
 E  OBJECTIVE  ELEM1    1.0
 E  OBJECTIVE  ELEM2    2.0
```

### GPS Complexity Levels

#### Level 1: Simple Separable
- Each variable appears in only one element
- No interaction terms
- Direct vectorization possible

#### Level 2: Partially Separable
- Variables may appear in multiple elements
- Limited interaction terms
- Requires careful index management

#### Level 3: Complex GPS
- Nested group structures
- Conditional element evaluation
- May need custom JAX primitives

## Vectorization Strategies by GPS Type

### Strategy A: Full Vectorization
For problems where all elements have the same structure:
```python
# All elements computed in parallel
element_values = vmap(element_func)(params, variables)
result = jnp.sum(element_values * weights)
```

### Strategy B: Grouped Vectorization
For problems with different element types:
```python
# Vectorize within element types
results = []
for element_type in self.element_types:
    mask = self.element_type_mask[element_type]
    values = vmap(self.element_funcs[element_type])(
        self.params[mask], y[self.var_indices[mask]]
    )
    results.append(values)
return jnp.concatenate(results)
```

### Strategy C: Sequential with Scan
For problems with dependencies:
```python
def scan_body(carry, element_data):
    accumulated, y = carry
    value = compute_element(element_data, y)
    return (accumulated + value, y), value

_, element_values = lax.scan(
    scan_body, (0.0, y), self.element_data
)
```

## Common GPS Implementation Patterns

### Pattern 1: Distance-Based
```python
# Common in facility location, clustering
distances = jnp.linalg.norm(
    points[:, None, :] - centers[None, :, :], 
    axis=2
)
objective = jnp.sum(jnp.min(distances, axis=1))
```

### Pattern 2: Product Terms
```python
# Common in bilinear problems
products = y[self.idx1] * y[self.idx2]
objective = jnp.sum(products * self.coeffs)
```

### Pattern 3: Trigonometric
```python
# Common in signal processing problems
phases = jnp.arange(n) * y[0]
amplitudes = y[1:n+1]
signal = jnp.sum(amplitudes * jnp.sin(phases))
```

### Pattern 4: Exponential
```python
# Common in fitting problems
exponentials = jnp.exp(-self.rates * y[0])
weighted = exponentials * y[1:] 
objective = jnp.sum((weighted - self.data) ** 2)
```

## GPS to JAX Translation Rules

| GPS Component | JAX Translation | Optimization |
|--------------|-----------------|--------------|
| Element sum | `jnp.sum` | Use `axis` parameter |
| Element product | `jnp.prod` | Consider log-sum-exp |
| Group aggregation | `scatter_add` or loop | Pre-sort if possible |
| Weighted sum | `jnp.dot` | Use BLAS when applicable |
| Conditional elements | `jnp.where` | Compute both branches |
| Sparse operations | COO format | Use JAX sparse ops |

## Performance Tips for GPS Problems

1. **Pre-compute index arrays** in `__init__`
2. **Batch similar computations** together
3. **Use JAX's XLA compilation** for repeated patterns
4. **Avoid Python loops** in objective/constraint evaluation
5. **Leverage symmetry** when present in GPS structure

## Debugging GPS Implementations

### Common Issues and Solutions

1. **Index mismatch**: SIF uses 1-based, Python 0-based
   - Solution: Subtract 1 from all SIF indices

2. **Group accumulation order**: Floating point precision
   - Solution: Sort elements by group before summation

3. **Missing elements**: Some elements may be implicit
   - Solution: Check SIF file comments and defaults

4. **Scaling issues**: Group/element scales not applied
   - Solution: Verify GSCALE and ESCALE parameters

## GPS Pattern Examples from CUTEst

### ROSENBR (Simple GPS)
```python
# Sum of (100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2)
def objective(self, y: ArrayLike) -> Scalar:
    return jnp.sum(
        100 * (y[1:] - y[:-1]**2)**2 + (1 - y[:-1])**2
    )
```

### DIXON3DQ (Quadratic GPS)
```python
# Quadratic with sparse structure
def objective(self, y: ArrayLike) -> Scalar:
    diag = (y - 1)**2
    off_diag = (y[:-1] - y[1:])**2
    return jnp.sum(diag) + jnp.sum(off_diag)
```

### ARWHEAD (Arrow-head GPS)
```python
# Special structure: interactions with first variable
def objective(self, y: ArrayLike) -> Scalar:
    tail_sum = jnp.sum((y[:-1]**2 + y[-1]**2)**2)
    head_terms = jnp.sum(y[:-1])
    return tail_sum - 4*head_terms + 3*(self.n-1)
```