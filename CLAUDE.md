# SIF2JAX Conversion Assistant

## Mission
Convert CUTEST problems to JAX implementations that match Fortran precision and are as performant as possible.

## Quick Reference
```bash
bash run_tests.sh --test-case "PROBLEM1,PROBLEM2"  # Test specific problems
bash run_tests.sh -k test_objective                 # Test specific aspect
ruff format . && ruff check .                       # Format and lint
```

## Workflow: Find → Implement → Test → Fix → Commit → Repeat

### 1. Find Next Problem
Check `missing_problems.md` for unchecked items `[] PROBLEMNAME` that are NOT imported in `sif2jax/__init__.py`

### 2. Implementation Priority
1. **AMPL**: `https://github.com/ampl/global-optimization/tree/master/cute` (lowercase.mod files)
2. **SIF**: For constants, bounds, starting points
3. **extra_info/**: Papers, documentation, screenshots
4. **References**: From SIF file headers

### 3. Implementation Rules
- **Name**: Use SIF name as class name (modify if invalid Python)
- **Metadata**: All references, authors, classification in docstring
- **Base Class**: Choose correct type:
  - `UnconstrainedProblem`: objective only
  - `BoundedProblem`: objective + bounds  
  - `ConstrainedProblem`: objective + constraints (+ bounds)
  - `NonlinearEquations`: residual functions
- **Types**: Never hard-code dtypes
- **Style**: Match existing code patterns, imports, conventions
- **Fields**: Declare all dataclass fields (Equinox.Module inheritance)

### 4. Testing Requirements
- **Container Required**: Tests need pycutest/Fortran libs
- **Test After EVERY Change**: Even minor edits
- **Batch Testing**: Full test suite will result in timeout in devcontainer, test problems individually or in small batches instead
- **5 Attempts Rule**: After 5 failed attempts, flag for human review:
  ```python
  # TODO: Human review needed
  # Attempts made: [list attempts]
  # Suspected issues: [your analysis]
  # Resources needed: [what would help]
  ```

### 5. Commit Process
- ✓ All tests pass
- ✓ Run pre-commit (NEVER use --no-verify)
- ✓ Fix all pre-commit issues
- ✓ Re-test if any changes made
- ✓ Success = clean pre-commit (no reformatting, no warnings)
- ✓ Commit after implementing a few problems, to keep commits compact

## Work Mode
- Add 20+ problems to TODO list
- Work systematically without stopping
- Re-read this prompt every 20 completed items

## PR Guidelines
Run `git diff main` and summarize the diff, not just latest commits.

## Performance Target
JAX implementation must be within 5x of Fortran runtime. For larger problems JAX is expected to be faster.
Avoid using for-loops, vectorise implementations wherever possible. 
If a sequential operation is needed, use a jax-native option such as a scan.