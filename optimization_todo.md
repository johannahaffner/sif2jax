# JAX Optimization TODO List

## Optimization Results

### Recently Optimized (2025-01-10)
- **CHNRSNBM** - Improved but still 6.19x obj slowdown (was not in original list) - Used vmap for vectorization
- **ENGVAL1** - Successfully optimized, now within threshold - Replaced for loops with vectorized operations
- **HELIX** - Successfully optimized, now within threshold (was 5.10x obj slowdown) - Simplified trigonometric computations
- **JENSMP** - Successfully optimized, now within threshold (was 5.04x combined slowdown) - Vectorized sum operations

## Problems Slower Than pycutest (Sorted by Priority)

### Critical Issues (>4x slowdown in any metric)
1. **HS56** - 9.74x obj slowdown - Check for inefficient sin operations
2. **ENGVAL2** - 8.65x obj slowdown - Review implementation
3. **DENSCHNA** - 5.47x combined slowdown - Investigate combined computation
4. **CHNROSNB** - 4.16x combined slowdown - Check chain rule complexity
7. **HILBERTB** - 4.07x grad slowdown - Matrix operations inefficiency
8. **HS11** - 4.05x obj slowdown - Small problem with division
9. **HS65** - 4.03x obj slowdown - Review implementation

### High Priority (3-4x slowdown)
10. **HS28** - 3.88x obj slowdown
11. **HS57** - 3.77x grad slowdown
12. **HS26** - 3.71x obj slowdown
13. **HS30** - 3.64x obj slowdown
14. **HS69** - 3.55x obj slowdown
15. **HS39** - 3.34x obj slowdown
16. **HS16** - 3.24x grad slowdown
17. **BT3** - 3.23x obj slowdown
18. **HS51** - 3.22x grad slowdown
19. **HS25** - 3.16x grad slowdown
20. **HS36** - 3.09x grad slowdown

### Medium Priority (2-3x slowdown)
21. **KIRBY2LS** - 2.97x obj slowdown
22. **CHWIRUT2LS** - 2.95x grad slowdown
23. **EGGCRATE** - 2.88x grad slowdown
24. **BT7** - 2.79x obj slowdown
25. **BT8** - 2.79x obj slowdown
26. **BROWNBS** - 2.62x obj slowdown
27. **DENSCHNB** - 2.55x obj slowdown
28. **ELATVIDU** - 2.51x obj slowdown
29. **HS45** - 2.46x combined slowdown
30. **HS55** - 2.43x obj slowdown

## Common Optimization Strategies

1. **For Small Problems (n < 10)**:
   - Consider batching operations
   - Reduce Python overhead
   - Use specialized implementations for common patterns

2. **For Trigonometric Operations**:
   - Review JAX's trig function implementations
   - Consider caching intermediate results
   - Use JAX-optimized formulations

3. **For Combined Operations**:
   - Ensure proper fusion of objective and gradient
   - Use JAX's value_and_grad where appropriate
   - Avoid redundant computations

4. **General Optimizations**:
   - Check for unnecessary array copies
   - Ensure proper use of JAX's vectorization
   - Review indexing operations for efficiency