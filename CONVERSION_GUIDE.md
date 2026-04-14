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

### 3. Common Pitfalls
- **1-based indexing**: SIF uses 1-based, Python uses 0-based
- **Parameter values**: Some SIF files have implicit parameters
- **Scaling factors**: May be hidden in GROUP definitions

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
| DO loop | slices / broadcasting | Prefer over vmap+arange |
| GROUP | Function composition | May need accumulation |
| ELEMENT | Vectorized function | Batch evaluate |
| IF-THEN | jnp.where / lax.cond | Avoid Python if |
| Sum | jnp.sum | Check axis parameter |
| Product | jnp.prod | Or reduce with * |
| x(i) | y[i-1] | Mind 0-based indexing |

## Performance: Avoiding Gather/Scatter in Objectives

JAX's `jnp.arange` + indexing produces `gather` ops whose reverse-mode AD
uses `scatter-add` — 2-5x slower than the `pad` VJP of slices. Patterns to
avoid and their replacements:

| Avoid | Use instead |
|-------|-------------|
| `y[jnp.arange(n-1)]` | `y[:n-1]` |
| `y[jnp.arange(n-1) + 1]` | `y[1:]` |
| `y[2*jnp.arange(s)]` | `y[:2*s:2]` |
| `y[3*jnp.arange(s) + k]` | `y[k:3*s+k:3]` |
| `vmap(f)(jnp.arange(n))` where f does `y[i]` | Rewrite f with slices |
| `res.at[::2].set(a)` | `jnp.stack([a, b], axis=1).flatten()` |
| `res.at[k:].add(v)` | `jnp.concatenate([zeros(k), v])` + add |
| `(jnp.arange(p)+1) % p` (cyclic) | `jnp.roll(arr, -1, axis=0)` |

For modular permutation indexing (`(k*i) % n`) that can't be expressed as
slices, keep `jnp.arange` and set `EAGER_CONSTANT_FOLDING=TRUE` so JAX folds
the index computation at trace time.

Run `tests/test_jaxpr.py` to verify objectives are gather/scatter-free.