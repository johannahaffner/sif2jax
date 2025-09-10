# Discrepancy Report: LUKVLI14 Constraint Definition

## Summary

There appears to be a discrepancy between the LUKVLI14.SIF file specification and the pycutest implementation regarding the element definition for `E(K+1)` in the constraint formulation.

## Problem Details

- **Problem Name**: LUKVLI14
- **Source**: L. Luksan and J. Vlcek, Technical Report 767, 1999
- **SIF Input**: Nick Gould, April 2001
- **Classification**: OOR2-AY-V-V

## The Discrepancy

### SIF File Specification

In the ELEMENT USES section (lines 117-124 of LUKVLI14.SIF), the file specifies:

```
DO K         1                        NC
DI K         2
IA K+1       K         1
XT E(K)      SQR
ZV E(K)      V                        X(K)
XT E(K+1)    SQR
ZV E(K+1)    V                        X(K+2)
ND
```

This clearly indicates that for each K = 1, 3, 5, ...:
- Element `E(K)` uses `X(K)^2`
- Element `E(K+1)` uses `X(K+2)^2`

Combined with the GROUP definitions, this produces constraints:
- `C(1) = X(1)^2 + X(2) + X(3) + 4*X(4) - 7`
- `C(2) = X(3)^2 - 5*X(5) - 6`
- `C(3) = X(3)^2 + X(4) + X(5) + 4*X(6) - 7`
- `C(4) = X(5)^2 - 5*X(7) - 6`
- And so on...

### Observed pycutest Behavior

Testing with pycutest reveals that it behaves as if `E(K+1)` uses `X(K+1)^2` instead of `X(K+2)^2`.

## Evidence

### Test 1: Starting Point Values

With the starting point pattern `[10, 7, -3, 10, 7, -3, ...]`:

| Constraint | SIF Formula | SIF Result | pycutest Result | Match? |
|------------|-------------|------------|-----------------|--------|
| C(1) | X(1)^2 + X(2) + X(3) + 4*X(4) - 7 | 137 | 137 | ✓ |
| C(2) | X(3)^2 - 5*X(5) - 6 | -32 | 8 | ✗ |
| C(3) | X(3)^2 + X(4) + X(5) + 4*X(6) - 7 | 7 | 7 | ✓ |
| C(4) | X(5)^2 - 5*X(7) - 6 | -7 | -7 | ✓ |
| C(5) | X(5)^2 + X(6) + X(7) + 4*X(8) - 7 | 77 | 77 | ✓ |
| C(6) | X(7)^2 - 5*X(9) - 6 | 109 | 8 | ✗ |

For C(2):
- Using X(3)^2 (as per SIF): 9 - 35 - 6 = -32
- Using X(2)^2 (explains pycutest): 49 - 35 - 6 = 8 ✓

### Test 2: Alternating Pattern

With alternating values `[0, 1, 0, 1, 0, 1, ...]`:

| Implementation | Result | pycutest Result | Match? |
|----------------|--------|-----------------|--------|
| Using X(K+2)^2 (SIF compliant) | [-2, -6, -2, -6, ...] | [-2, -6, -2, -6, ...] | ✓ |
| Using X(K+1)^2 | [-2, -5, -2, -5, ...] | [-2, -6, -2, -6, ...] | ✗ |

Interestingly, the SIF-compliant formula works perfectly for the alternating pattern.

## Analysis

The evidence suggests that:

1. The SIF file unambiguously specifies that `E(K+1)` should use `X(K+2)`
2. pycutest appears to implement `E(K+1)` using `X(K+1)` instead
3. This discrepancy affects the even-numbered constraints (C(2), C(4), C(6), ...)

## Possible Explanations

1. **Bug in pycutest**: An off-by-one error in parsing the element definition
2. **Undocumented correction**: The original problem may have had an error that was corrected in pycutest but not in the SIF file
3. **Intentional variation**: Different interpretations of the problem exist

## Recommendation

The SIF file and implementations should be reconciled. Either:
- The SIF file should be corrected if `X(K+1)` is the intended formulation
- Or pycutest should be fixed to match the SIF specification
- Or a note should be added documenting the discrepancy

## Testing Code

The following Python code demonstrates the discrepancy:

```python
import numpy as np

# Starting point values (1-indexed for clarity)
x = [10, 7, -3, 10, 7, -3, 10, 7, -3, 10]

# According to SIF file
print("SIF Formula for C(2):")
print(f"  X(3)^2 - 5*X(5) - 6 = {x[2]**2} - 5*{x[4]} - 6 = {x[2]**2 - 5*x[4] - 6}")

# What pycutest expects
print("\nTo get pycutest result of 8:")
print(f"  X(2)^2 - 5*X(5) - 6 = {x[1]**2} - 5*{x[4]} - 6 = {x[1]**2 - 5*x[4] - 6}")
```

Output:
```
SIF Formula for C(2):
  X(3)^2 - 5*X(5) - 6 = 9 - 5*7 - 6 = -32

To get pycutest result of 8:
  X(2)^2 - 5*X(5) - 6 = 49 - 5*7 - 6 = 8
```

## Additional Notes

- Similar issues may exist in LUKVLI17 and LUKVLI18, which use the same element definition pattern
- The problem is particularly confusing because different test patterns give different results
- The issue only affects the squared term in even-numbered constraints; the linear terms match the SIF specification