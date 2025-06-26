# PyCUTEst Constraint Conventions

Based on empirical testing of various problem types from the CUTEst collection, here's how pycutest represents different constraint types using the `.cl` (lower bounds) and `.cu` (upper bounds) attributes:

## Constraint Representation

PyCUTEst uses a unified representation where all constraints are expressed as:
```
cl[i] <= c[i](x) <= cu[i]
```

Where:
- `cl[i]` is the lower bound for constraint i
- `cu[i]` is the upper bound for constraint i
- `c[i](x)` is the constraint function value at point x

## Constraint Types

### 1. Equality Constraints: `h(x) = 0`
- `cl = 0.0`
- `cu = 0.0`
- `is_eq_cons = True`
- Example: BT1, HS14 (constraint 2)

### 2. Greater-than-or-equal (G-type) Inequalities: `g(x) >= b`
- `cl = b` (the bound value, often 0.0)
- `cu = 1e+20` (effectively +∞)
- `is_eq_cons = False`
- Example: HS10 has `g(x) >= 0` represented as `0 <= c(x) <= 1e+20`

### 3. Less-than-or-equal (L-type) Inequalities: `g(x) <= b`
- `cl = -1e+20` (effectively -∞)
- `cu = b` (the bound value, often 0.0)
- `is_eq_cons = False`
- Example: HS64 has `g(x) <= 0` represented as `-1e+20 <= c(x) <= 0`

### 4. Range Constraints: `a <= g(x) <= b`
- `cl = a` (finite lower bound)
- `cu = b` (finite upper bound)
- `is_eq_cons = False`
- Example: HS83 has constraints like `0 <= c(x) <= 92`

## Important Notes

1. **Infinity values**: PyCUTEst uses `±1e+20` to represent ±∞ in constraint bounds.

2. **Standard form**: If your optimization package expects constraints in the form `g(x) >= 0`, you'll need to transform:
   - For G-type: Use `c(x) - cl[i] >= 0`
   - For L-type: Use `-c(x) + cu[i] >= 0`
   - For range: Split into two constraints or handle as `cl[i] <= c(x) <= cu[i]`
   - For equality: Keep as `c(x) = 0`

3. **Consistency**: The `is_eq_cons` flag reliably identifies equality constraints, which always have `cl = cu = 0`.

## Test Results Summary

| Problem | Type | cl | cu | is_eq_cons | Original Form |
|---------|------|----|----|------------|---------------|
| BT1 | Equality | 0.0 | 0.0 | True | h(x) = 0 |
| HS10 | G-type | 0.0 | 1e+20 | False | g(x) >= 0 |
| HS64 | L-type | -1e+20 | 0.0 | False | g(x) <= 0 |
| HS83 | Range | [0,0,0] | [92,20,5] | [False,False,False] | 0 <= g(x) <= bounds |
| HS76 | Mixed L&G | varies | varies | False | Mixed constraints |
| HS14 | Mixed Eq&G | varies | varies | [False,True] | Mixed constraints |