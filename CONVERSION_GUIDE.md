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

### Near-dense COO → dense matmul
Some problems store a quadratic form as COO sparse (Q_row, Q_col, Q_val)
even when the matrix is nearly dense. If density is >~50%, consider converting
to dense (this threshold is a rule of thumb — even sparser matrices may benefit
since matmul's AD is much cheaper than gather's):

```python
# Bad: two gathers per evaluation, scatter-add in VJP
quad = jnp.sum(Q_val * y[Q_row] * y[Q_col])

# Good: dense matmul, no gathers in jaxpr
Q = jnp.zeros((n, n)).at[Q_row, Q_col].add(Q_val)
quad = y @ Q @ y
```

The `.at[].add()` uses scatter, but since Q_row/Q_col/Q_val are all constants,
`eager_constant_folding` materializes the dense Q at trace time — the final
jaxpr contains only `dot_general` (matmul) with no gather or scatter.

## Performance: Keeping the AD Graph Compact

Second-order transforms (Hessians, HVPs) are sensitive to the number of
intermediate values in the computation graph. Patterns to keep it small:

### Avoid meshgrid for coefficient arrays
`jnp.meshgrid` creates `(p, l, s)` intermediates. With `eager_constant_folding`,
these get materialized as large closure constants transferred to device each call.

```python
# Bad: creates (4, 4, s) coefficient arrays
L, Q = jnp.meshgrid(l_vals, q_vals, indexing="ij")
result = (L**2 * Q)[:,:,None] * f(x_vals)  # (4, 4, s)

# Good: keep small vectors, use dot products
l2 = l_vals**2                              # (4,)
result = l2[:, None] * (q_vals @ f(x_q))   # (4, s)
```

### Factor separable sums before broadcasting
When coefficients factor as `a[i,j] = g(i) * h(j)`, reduce over the inner
dimension first via dot product, then scale:

```python
# Bad: (l, q, s) intermediate
total = a_coeff[:, :, None] * sin_vals[None, :, :]  # (4, 4, s)
result = jnp.sum(total, axis=1)                     # (4, s)

# Good: reduce q first, then broadcast l
q_sum = h_coeffs @ sin_vals    # (4,) @ (4, s) -> (s,)
result = g_coeffs[:, None] * q_sum[None, :]  # (4, s)
```

### Keep expensive ops batched
One call on a stacked array produces fewer AD nodes than multiple calls
on slices. This matters most for Hessian computation.

```python
# Bad for Hessians: 4 separate sin ops, 4 intermediate buffers
s1, s2, s3, s4 = jnp.sin(x1), jnp.sin(x2), jnp.sin(x3), jnp.sin(x4)

# Good: 1 sin op on stacked (4, s) array, 1 intermediate buffer
x_stacked = jnp.stack([x1, x2, x3, x4])  # (4, s)
sin_all = jnp.sin(x_stacked)              # (4, s)
```