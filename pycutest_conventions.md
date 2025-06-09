# PyCUTEst Constraint Formulation Conventions

This document outlines the constraint formulation conventions used by pycutest when importing problems from SIF files, based on empirical observations made while fixing test problems in the sif2jax project.

## Key Discoveries

### 1. Constraint Types and Sign Conventions

**General vs Specific Sign Handling:**
- For **inequality constraints**, pycutest returns the raw constraint value as defined in the SIF file
- The traditional convention of `g(x) ≤ 0` is NOT automatically applied by pycutest
- Constraint types in SIF files:
  - `G` (Greater than or equal): `constraint_expression ≥ constant`
  - `L` (Less than or equal): `constraint_expression ≤ constant`
  - `E` (Equality): `constraint_expression = constant`

**Example from HS93:**
```python
# SIF file defines:
# C1 (G type): 0.001 * x1*x2*x3*x4*x5*x6 >= 2.07
# C2 (L type): 0.00062*OE3 + 0.00058*OE4 <= 1

# pycutest returns:
# C1: 0.001 * x1 * x2 * x3 * x4 * x5 * x6 - 2.07  (raw value)
# C2: 0.00062 * OE3 + 0.00058 * OE4 - 1.0         (raw value)
```

### 2. Ranged Constraints

**Single Constraint Representation:**
- Ranged constraints (with both lower and upper bounds) are represented as **single constraints** in pycutest
- They are NOT split into separate upper and lower bound constraints

**Example from HS116:**
```python
# PDF/AMPL formulation: 50 ≤ f(x) ≤ 250
# SIF file: Handled as ranged constraint C4
# pycutest: Returns as single constraint (not two)
```

### 3. Ranged Constraint Shifting

**Important:** For ranged constraints, pycutest shifts the constraint values by the negative of their constants to normalize the range to start at 0.

**Formula:**
If SIF defines: `lower_bound ≤ constraint_expression ≤ upper_bound`

Where the constraint in SIF is written as:
- Constraint group: `expression`
- Constant: `lower_bound` (specified in CONSTANTS section)
- Range: `upper_bound - lower_bound` (specified in RANGES section)

Then pycutest returns: `constraint_expression - lower_bound`

**Example from HS83:**
```python
# SIF file defines:
# C1: -85.334407 + 0.0056858*x2*x5 + 0.0006262*x1*x4 - 0.0022053*x3*x5
# With constant CS1 = -85.334407 and range 92

# pycutest returns:
# (0.0056858*x2*x5 + 0.0006262*x1*x4 - 0.0022053*x3*x5) - (-85.334407)
# = 0.0056858*x2*x5 + 0.0006262*x1*x4 - 0.0022053*x3*x5 + 85.334407
```

### 4. Constraint Ordering

**SIF File Order:**
- pycutest follows the exact constraint ordering from the SIF file
- This ordering may differ from the problem's documentation (PDF/AMPL)

**Example from HS118:**
```python
# SIF file defines constraints in alternating pattern:
# A(1), C(1), B(1), A(2), C(2), B(2), ...
# NOT grouped as A(1-4), B(1-4), C(1-4) as might be expected
```

### 5. Problem Classification

**Constraint-based Classification:**
- Problems with **zero constraints** (only variable bounds) belong to `AbstractBoundedMinimisation`
- Problems with constraints belong to `AbstractConstrainedMinimisation`
- pycutest's `m` attribute indicates the number of constraints

**Example from HS110:**
```python
# Despite being listed as HS110 in constrained problems
# pycutest reports m=0 (zero constraints)
# Therefore belongs to bounded minimization category
```

## Implementation Guidelines for Compatibility Wrapper

Based on these observations, a compatibility wrapper should:

1. **Preserve Raw Constraint Values**: Return constraint values exactly as pycutest does, without applying sign conventions

2. **Handle Ranged Constraints**: 
   - Detect ranged constraints from SIF file metadata
   - Apply appropriate shifting: `raw_expression + constant` for normalized representation

3. **Maintain Constraint Order**: Parse SIF files to determine exact constraint ordering

4. **Classify Problems Correctly**: Check number of constraints to determine problem category

5. **Document Constraint Types**: Clearly indicate whether each constraint is:
   - One-sided inequality (G or L type)
   - Equality (E type)
   - Ranged inequality (has both CONSTANTS and RANGES)

## Testing Strategy

To verify compatibility:

1. Compare constraint values at initial point
2. Test constraint gradients and Hessians
3. Verify constraint bounds interpretation
4. Check problem classification matches pycutest

## Common Pitfalls to Avoid

1. **Don't assume g(x) ≤ 0 convention**: pycutest doesn't automatically transform constraints
2. **Don't split ranged constraints**: Keep them as single constraints
3. **Don't ignore constant shifting**: Essential for ranged constraints
4. **Don't assume documentation ordering**: Always check SIF file