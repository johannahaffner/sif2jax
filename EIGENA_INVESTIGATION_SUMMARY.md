# EIGENA Series Investigation Summary

## Overview
The EIGENA family of problems (EIGENA, EIGENAU, EIGENA2, EIGENACO) implements symmetric eigenvalue decomposition as various optimization formulations. All problems solve for orthogonal matrix Q and diagonal matrix D such that A = Q^T D Q, where A is a diagonal matrix with eigenvalues 1, ..., N (N=50).

## Problems Addressed

### ✅ **Initial Values** - FIXED
**Issue**: Initial values didn't match pycutest behavior  
**Root Cause**: pycutest uses complex pattern instead of SIF specification  
**Solution**: Implemented pycutest's exact pattern:
- Eigenvalues: D[0]=D[1]=1.0, rest are 0
- Eigenvectors: Complex sparse pattern with special handling for rows 24, 25-48, and 49

### ✅ **Bounds** - FIXED  
**Issue**: Missing bounds caused test failures  
**Root Cause**: SIF comment "nonnegative eigenvalues" interpreted as bounds by pycutest  
**Solution**: Added bounds property with lower=0, upper=∞ for all variables

### ✅ **Constraint Ordering** - FIXED
**Issue**: Constraint ordering mismatch causing element-level discrepancies  
**Root Cause**: Tried multiple orderings (interleaved, orthogonality-first, etc.)  
**Solution**: Found eigen constraints first, then orthogonality constraints works best

### ❌ **Constraint Values** - UNSOLVED
**Issue**: Systematic 50.0, 49.0, etc. discrepancies in constraint values  
**Root Cause**: Fundamental difference between sif2jax mathematical formulation and pycutest's SIF interpretation  
**Impact**: Constraints mathematically correct but don't match pycutest expectations

## Technical Details

### Mathematical Formulation (sif2jax)
```
Eigen equations: Q^T D Q[i,j] - A[i,j] = 0
Orthogonality: Q^T Q[i,j] - I[i,j] = 0
```

### Constraint Value Pattern
- E(49,49): Our value = -50.0, pycutest = 0.0 (difference = 50.0)  
- E(48,48): Our value = -49.0, pycutest = 0.0 (difference = 49.0)
- Pattern suggests pycutest handles diagonal eigen-constraints differently

### Test Results (Before vs After)
- **EIGENA**: ~16/20 → ~18/21 tests passing
- **Bounds tests**: All now pass 
- **Initial value tests**: All now pass
- **Constraint tests**: Still fail due to systematic value differences

## Attempts Made

1. **Special handling for unused eigenvalues**: Set E(k,k)=0 for k≥2 - reduced error but created new discrepancies
2. **Different constraint orderings**: Interleaved, orthogonality-first, SIF loop order - found optimal ordering but values still differ
3. **Investigation of SIF GROUPS/CONSTANTS sections**: Analyzed how pycutest might interpret constraint formulation differently

## Suspected Root Cause

The core issue appears to be that **pycutest interprets the SIF constraint formulation differently** than our mathematical implementation. The systematic nature of the discrepancies (exactly matching eigenvalue magnitudes) suggests pycutest may:

1. Use different constant terms in constraint equations
2. Exclude certain constraints that involve unused eigenvalues  
3. Apply some preprocessing to the SIF GROUPS/CONSTANTS sections
4. Have undocumented special handling for eigenvalue problems

## Human Review Requirements

**Expertise Needed:**
- Deep knowledge of pycutest's SIF file parsing internals
- Understanding of how GROUPS and CONSTANTS sections interact  
- Familiarity with pycutest's constraint generation process
- Possibly consultation with pycutest maintainers

**Investigation Steps:**
1. Compare pycutest's generated constraints with SIF file structure element-by-element
2. Analyze pycutest source code for special eigenvalue problem handling
3. Debug pycutest's constraint generation process for EIGENA specifically
4. Consider whether pycutest has undocumented eigenvalue constraint conventions

## Status: Marked for Human Review

All EIGENA series problems are commented out from the codebase:
- `EIGENA` (nonlinear equations)
- `EIGENAU` (nonlinear equations) 
- `EIGENA2` (quadratic problems)
- `EIGENACO` (constrained minimization)

**Verification**: `python -c "import sif2jax"` succeeds, library contains 599 problems (down from 603).