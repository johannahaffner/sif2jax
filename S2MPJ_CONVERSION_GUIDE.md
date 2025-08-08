# S2MPJ-Inspired SIF2JAX Conversion Guide

## Overview
This guide applies insights from the S2MPJ paper (arXiv:2407.07812) to improve SIF→JAX conversions.

## Key Concepts from S2MPJ

### 1. Group-Partially-Separable (GPS) Structure
Most CUTEst problems follow a GPS pattern:
```
f(x) = Σ(group_functions) + quadratic_term
```

In JAX, vectorize these patterns:
- Identify repeating group structures
- Use `vmap` for element-wise operations
- Batch similar computations

### 2. Problem Components to Identify

#### From SIF File Headers
- **GROUPS**: Define how objective/constraints are structured
- **ELEMENTS**: Nonlinear components that get combined
- **VARIABLES**: Problem dimensions and bounds
- **START POINT**: Initial values (critical for validation)

#### Pattern Recognition
Look for these GPS patterns in SIF files:
```
GROUP TYPE <name>
ELEMENT TYPE <name>
```
These indicate reusable computation patterns perfect for JAX vectorization.

## Conversion Workflow

### Step 1: Analyze SIF Structure
```bash
# Check SIF file for GPS patterns
grep -E "GROUP TYPE|ELEMENT TYPE|DO " archive/mastsif/PROBLEM.SIF
```

### Step 2: Identify Vectorization Opportunities
- **DO loops** → `jnp.arange` + vectorized ops
- **Repeated GROUPS** → `vmap` over group indices
- **Element evaluations** → Batch compute all elements

### Step 3: Implementation Patterns

#### Pattern A: Separable Sum
```python
# SIF: Sum of element functions
def objective(self, y: ArrayLike) -> Scalar:
    # Vectorized element evaluation
    elements = self._compute_elements(y)  # Shape: (n_elements,)
    return jnp.sum(elements)
    
def _compute_elements(self, y: ArrayLike) -> Array:
    # Example: squared differences
    indices = self.element_indices  # Pre-computed in __init__
    values = y[indices]
    return (values - self.targets) ** 2
```

#### Pattern B: Group Structure
```python
# SIF: Groups with weighted elements
def objective(self, y: ArrayLike) -> Scalar:
    # Compute nonlinear elements
    elements = self._compute_elements(y)
    
    # Vectorized group aggregation using scatter
    # Pre-computed in __init__: 
    # self.group_indices: which group each element belongs to
    # self.weights: element weights
    weighted_elements = elements * self.weights
    
    # Sum elements by group using segment_sum or scatter
    group_values = jax.ops.segment_sum(
        weighted_elements, 
        self.group_indices,
        num_segments=self.n_groups
    )
    
    # Apply group scaling
    scaled_groups = group_values * self.group_scales
    
    # Add linear/quadratic terms if present
    return jnp.sum(scaled_groups) + self._quadratic_term(y)
```

#### Pattern C: Constraint Handling
```python
def constraints(self, y: ArrayLike) -> Array:
    # Vectorize constraint evaluation
    # Similar GPS structure as objective
    return self._compute_constraint_groups(y)
```

### Step 4: Validation Against pycutest

Validate directly against the native Fortran implementation via pycutest:
1. Compare objective values at x0
2. Compare gradient values  
3. Check constraint evaluations

```python
# Validation template
def expected_result(self) -> Scalar:
    """Compare with pycutest Fortran result"""
    # From pycutest evaluation at x0
    # This is computed during test generation
    return expected_value
```

Note: Our test suite automatically validates against pycutest, which provides the ground truth from the original Fortran implementations.

## Common SIF→JAX Mappings

### 1. Index Patterns
```
SIF: DO I 1 N
JAX: i = jnp.arange(n)
```

### 2. Conditional Logic
```
SIF: IF-THEN blocks
JAX: jnp.where or lax.cond
```

### 3. Summations
```
SIF: Sum over groups/elements
JAX: jnp.sum with appropriate axis
```

### 4. Products
```
SIF: Product terms
JAX: jnp.prod or sequential multiplication
```

## Performance Optimization

### Vectorization Priority
1. **Always vectorize** element/group evaluations
2. **Avoid Python loops** - use JAX primitives
3. **Pre-compute** indices and masks in `__init__`
4. **Batch operations** where possible

### Memory Efficiency
- Use views instead of copies
- Leverage JAX's lazy evaluation
- Pre-allocate arrays when size is known

## Debugging Tips

### 1. Dimension Mismatches
- Print shapes at each step during development
- Verify index ranges match SIF specifications

### 2. Numerical Precision
- S2MPJ achieves <10^-14 relative error
- If larger errors, check:
  - Index offsetting (0-based vs 1-based)
  - Accumulation order for sums
  - Floating point associativity

### 3. Common Pitfalls
- **1-based indexing**: SIF uses 1-based, Python uses 0-based
- **Parameter values**: Some SIF files have implicit parameters
- **Scaling factors**: May be hidden in GROUP definitions

## Testing Protocol

### Required Tests (Automatic via pycutest)
1. **Objective value** at x0 - compared to Fortran
2. **Gradient** computation - validated against Fortran gradients
3. **Constraint values** - checked against Fortran constraints
4. **Bounds checking** - verified from SIF specifications

### Test Command
```bash
# Test single problem against pycutest
bash run_tests.sh --test-case "PROBLEM"

# Test with local compilation check
bash run_tests.sh --test-case "PROBLEM" --local-tests
```

### Expected Precision vs Fortran
- Relative error < 1e-10 for most problems
- Some ill-conditioned problems may have larger errors
- Tests automatically compare against pycutest (Fortran) results
- Document any precision issues in comments

## References for Specific Patterns

### Least Squares Problems
- Often have GPS structure with squared residuals
- Look for: `(observation - model(x, params))^2`
- Vectorize over observations

### Network Problems
- Node/edge structure maps well to JAX arrays
- Use adjacency matrix representations
- Vectorize flow computations

### Discretized PDEs
- Grid structure → reshape to 2D/3D arrays
- Finite difference stencils → convolutions
- Boundary conditions → padding or masking

## Quick Reference Card

| SIF Construct | JAX Equivalent | Notes |
|--------------|----------------|--------|
| DO loop | jnp.arange + vmap | Vectorize when possible |
| GROUP | Function composition | May need accumulation |
| ELEMENT | Vectorized function | Batch evaluate |
| IF-THEN | jnp.where / lax.cond | Avoid Python if |
| Sum | jnp.sum | Check axis parameter |
| Product | jnp.prod | Or reduce with * |
| x(i) | y[i-1] | Mind 0-based indexing |

## Validation Checklist

- [ ] Objective matches pycutest at x0 (< 1e-10 relative error)
- [ ] Gradient matches pycutest (validated by test suite)
- [ ] Constraints match pycutest values
- [ ] Bounds match SIF specifications
- [ ] No Python loops in hot paths
- [ ] Pre-computation done in __init__
- [ ] Tests pass against pycutest
- [ ] Performance acceptable (< 5x Fortran via pycutest)

## When to Flag for Review

After 5 attempts, if issues persist:
1. Document attempted approaches
2. Note specific error patterns
3. Compare pycutest values at multiple points
4. Consider if problem needs special numerics
5. Add TODO comment with findings and pycutest comparison