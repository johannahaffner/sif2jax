# Deep Understanding of Group-Partially-Separable Structure

## The GPS Fundamental Insight

GPS structure means the objective/constraints can be written as:
```
f(x) = Σ_g α_g * G_g(Σ_e β_ge * E_e(x))
```

Where:
- **Groups (G_g)**: Outer functions that aggregate element values
- **Elements (E_e)**: Inner functions of subsets of variables
- **α_g, β_ge**: Scaling factors

This structure is KEY because:
1. Most real-world optimization problems naturally have this form
2. It enables massive parallelization
3. Sparsity patterns become clear
4. Derivatives have special structure

## How to Read SIF Files for GPS Structure

### Step 1: Understand the Sections

```sif
VARIABLES     # Define problem variables
GROUPS        # Define how elements combine (the G_g functions)
CONSTANTS     # Set numerical values
BOUNDS        # Variable bounds
START POINT   # Initial values
ELEMENT TYPE  # Templates for element functions (the E_e)
ELEMENT USES  # Assign elements to groups with weights
GROUP TYPE    # Templates for group functions
GROUP USES    # Apply group types to specific groups
```

### Step 2: Trace the GPS Flow

1. **Elements** compute values from variables
2. **Groups** aggregate element values
3. **Objective** sums group values

Example trace through ROSENBR:
```sif
ELEMENT TYPE EV 2PR        # Element: 2-variable product
 REAL    ALPHA
 VARIABLES V1 V2
 FORMULA  ALPHA * V1 * V2  # E_e(x) = α * x_i * x_j

GROUP USES
 T  'DEFAULT' SQUARES      # Group: squares its input
 
ELEMENT USES
 T  OBJ      2PR  1.0      # Group OBJ uses element 2PR with weight 1.0
```

## GPS Pattern Recognition in Practice

### Pattern 1: Sum of Separable Functions
**SIF Signature:**
```sif
DO I 1 N
 X  X(I)
 ND

DO I 1 N
 XE E(I)   X(I)   1.0
 ND
```

**JAX Translation:**
```python
# Each element depends on single variable
element_values = element_func(y)  # Vectorized over all variables
return jnp.sum(element_values)
```

### Pattern 2: Pairwise Interactions
**SIF Signature:**
```sif
DO I 1 N-1
 XE COUPLE(I)  X(I)  1.0
 XE COUPLE(I)  X(I+1) -1.0
 ND
```

**JAX Translation:**
```python
# Adjacent variable interactions
diffs = y[1:] - y[:-1]
return jnp.sum(func(diffs))
```

### Pattern 3: Sparse Quadratic
**SIF Signature:**
```sif
ELEMENT TYPE EV PROD
 VARIABLES X Y
 FORMULA X * Y

DO I 1 N
 DO J I+1 N
  XE Q(I,J)  X(I) 1.0
  XE Q(I,J)  X(J) 1.0
 ND
```

**JAX Translation:**
```python
# Sparse quadratic terms
i, j = self.quad_indices  # Pre-computed
products = y[i] * y[j]
return jnp.sum(products * self.quad_coeffs)
```

## Critical GPS Concepts

### 1. Element Variables vs Global Variables
- **Element variables**: Local names in ELEMENT TYPE
- **Global variables**: Actual problem variables
- Mapping happens in ELEMENT USES

### 2. Group Types
- **N (Nonlinear)**: General nonlinear function
- **L (Linear)**: Linear combination of elements
- **E (Equality)**: Constraint group
- **G/H (Greater/Less)**: Inequality constraint groups

### 3. Scaling Hierarchy
```
Total = Σ (group_scale * group_func(Σ (element_weight * element_func)))
```

## Advanced GPS Patterns

### Nested GPS Structure
Some problems have groups of groups:
```python
# Level 1: Elements
element_vals = compute_elements(y)

# Level 2: Sub-groups (vectorized aggregation)
subgroup_vals = jax.ops.segment_sum(
    element_vals,
    self.element_to_subgroup,
    num_segments=n_subgroups
)

# Level 3: Main groups (vectorized combination)
group_vals = jax.ops.segment_sum(
    subgroup_vals,
    self.subgroup_to_group,
    num_segments=n_groups
)

return jnp.sum(group_vals)
```

### Conditional GPS
Groups that activate based on conditions:
```python
# Elements always computed
element_vals = compute_elements(y)

# Groups conditionally aggregate
active_groups = y[0] > 0  # Example condition
group_vals = jnp.where(
    active_groups,
    aggregate_active(element_vals),
    aggregate_inactive(element_vals)
)
```

## GPS Analysis Tool

### Quick SIF GPS Analyzer
```python
def analyze_gps_structure(sif_file):
    """Extract GPS structure from SIF file"""
    
    with open(sif_file) as f:
        content = f.read()
    
    # Extract key sections
    groups = re.findall(r'GROUPS(.*?)(?=\n[A-Z])', content, re.DOTALL)
    elements = re.findall(r'ELEMENT TYPE(.*?)(?=\n[A-Z])', content, re.DOTALL)
    element_uses = re.findall(r'ELEMENT USES(.*?)(?=\n[A-Z])', content, re.DOTALL)
    
    # Count structures
    n_groups = len(re.findall(r'^\s*[NELGH]\s+', groups[0], re.MULTILINE))
    n_element_types = len(re.findall(r'^\s*T\s+', elements[0], re.MULTILINE))
    n_element_instances = len(re.findall(r'^\s*T\s+', element_uses[0], re.MULTILINE))
    
    # Identify patterns
    has_do_loops = 'DO ' in content
    has_conditionals = 'IF ' in content
    has_products = 'PROD' in content.upper()
    has_squares = 'SQR' in content.upper() or 'SQUARE' in content.upper()
    
    return {
        'n_groups': n_groups,
        'n_element_types': n_element_types,
        'n_element_instances': n_element_instances,
        'has_loops': has_do_loops,
        'has_conditionals': has_conditionals,
        'likely_pattern': guess_pattern(content)
    }

def guess_pattern(content):
    """Guess the GPS pattern from SIF content"""
    if 'DO I' in content and 'I+1' in content:
        return 'pairwise_interaction'
    elif 'SQUARE' in content and 'RESID' in content:
        return 'least_squares'
    elif 'PROD' in content and 'BILINEAR' in content:
        return 'bilinear'
    elif 'EXP' in content or 'LOG' in content:
        return 'transcendental'
    elif 'SIN' in content or 'COS' in content:
        return 'trigonometric'
    else:
        return 'general_nonlinear'
```

## GPS Implementation Checklist

When implementing a GPS problem:

1. **Identify Element Types**
   - [ ] List all ELEMENT TYPE definitions
   - [ ] Understand their mathematical functions
   - [ ] Note variable dependencies

2. **Map Element Instances**
   - [ ] Track which variables each element uses
   - [ ] Record element weights
   - [ ] Build element-to-group mapping

3. **Understand Group Structure**
   - [ ] Identify group types (N, L, E, etc.)
   - [ ] Map group scaling factors
   - [ ] Determine aggregation method

4. **Vectorization Strategy**
   - [ ] Can elements be computed in parallel?
   - [ ] Are there similar element types to batch?
   - [ ] Can groups be aggregated efficiently?

5. **Implementation**
   - [ ] Pre-compute all index arrays
   - [ ] Implement element functions
   - [ ] Implement group aggregation
   - [ ] Combine for final objective

## Example: Analyzing ROSENBR GPS Structure

```python
# ROSENBR has this GPS structure:
# f(x) = Σ[100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

# Elements:
# E1_i = x_{i+1} - x_i^2  (residual elements)
# E2_i = 1 - x_i          (linear elements)

# Groups:
# G1_i = 100 * E1_i^2     (squared residuals)
# G2_i = E2_i^2           (squared linear terms)

# JAX Implementation leveraging GPS:
def objective(self, y):
    # Compute elements
    e1 = y[1:] - y[:-1]**2  # Residual elements
    e2 = 1 - y[:-1]         # Linear elements
    
    # Apply group functions
    g1 = 100 * e1**2        # Squared with scaling
    g2 = e2**2              # Squared
    
    # Sum groups
    return jnp.sum(g1) + jnp.sum(g2)
```

## Key Insight for Pattern Matching

When you see these patterns in SIF:

| SIF Pattern | GPS Interpretation | JAX Strategy |
|------------|-------------------|--------------|
| `DO I 1 N` with similar elements | Separable structure | Single `vmap` over all |
| Nested `DO` loops | Grid or interaction structure | Reshape and use 2D ops |
| `ELEMENT TYPE` with 2+ variables | Coupling between variables | Index arrays for vectorization |
| Multiple `GROUP TYPE` | Different aggregation functions | Separate computation paths |
| `IF-THEN` in elements | Conditional GPS | `jnp.where` or `lax.cond` |

## Summary: Why GPS Matters

1. **Efficiency**: GPS reveals natural parallelization
2. **Sparsity**: Shows which variables interact
3. **Structure**: Guides vectorization strategy
4. **Debugging**: Helps validate against SIF intent
5. **Performance**: Enables optimal JAX compilation

Always start by understanding the GPS structure - it's the key to efficient JAX implementation!