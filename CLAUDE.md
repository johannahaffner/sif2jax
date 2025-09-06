# SIF2JAX Conversion Assistant

## Mission
Convert CUTEST problems to JAX implementations that match Fortran precision and are as performant as possible.

## Quick Reference

```bash
# Test specific problems (only runs tests/test_problem.py)
sudo bash run_tests.sh tests/test_problem.py --test-case "PROBLEM1,PROBLEM2"

# Run FULL test suite for a problem (all test modules including compilation tests)
sudo bash run_tests.sh --test-case "PROBLEM1" --local-tests

# Format and lint
ruff format . && ruff check .
```

**Important:** Specifying `tests/test_problem.py` limits testing to ONLY that module. To run the complete test suite, omit the file path and use just `sudo bash run_tests.sh --test-case "PROBLEM1"`.

(The test script may have to be run with `sudo bash` in the container.)

## Workflow: Implement → Test → Fix → Commit → Repeat

### 1. Overall goal
Problems only count as implemented if they pass ALL tests in the full test suite. 
- Quick validation: `sudo bash run_tests.sh tests/test_problem.py --test-case "PROBLEM1"` 
- Full validation: `sudo bash run_tests.sh --test-case "PROBLEM1" --local-tests`
The tests are designed to be very informative, and can guide you toward a working implementation.

### 2. Implementation Priority
SIF problems have a group-separable structure. Identifying this structure helps to identify opportunities for vectorisation and batched operations, as well as to `divide and conquer` complex problems.
You can look up some guiding principles in the CONVERSION_GUIDE file.

**Background on partially separable functions:** Griewank, A., Toint, Ph.L.: "On the unconstrained optimization of partially separable functions" in Powell, M.J.D. (ed.) Nonlinear Optimization 1981, pp. 301–312, Academic Press, London (1982). This foundational paper introduces the concept of functions that can be expressed as sums of element functions, enabling efficient derivative computation through the chain rule on separable components.

Here are the sources relevant to problem implementations:
1. **SIF Files**: `archive/mastsif/` folder - Original SIF problem definitions (PRIMARY SOURCE). Look in this folder whenever you are asked to find problems of any series - e.g. search for "TRO" in this folder to find problems from the "TRO" series of problems.
2. **AMPL**: `https://github.com/ampl/global-optimization/tree/master/cute` (lowercase.mod files)
3. **References**: From SIF file headers

### 3. Implementation Rules
- **Name**: Use SIF name as class name (modify if invalid Python)
- **Metadata**: All references, authors, classification in docstring
- **Base Class**: Choose correct type:
  - `AbstractUnconstrainedMinimisation`: objective only
  - `AbstractBoundedMinimisation`: objective + bounds  
  - `AbstractConstrainedMinimisation`: objective + constraints (+ bounds)
  - `AbstractConstrainedQuadraticProblem`: objective + constraints (+ bounds). This is
    a subclass of `AbstractConstrainedMinimisation` with no changes to the interface.
  - `AbstractNonlinearEquations`: provides default constant objective that may be 
    overridden; feasibility problem with constraints
    For guidance on problem classification, see: https://ralna.github.io/SIFDecode/html/classification/
- **Vectorization First**: For problems with n > 200 dimensions, ALWAYS write vectorized implementations from the start. Tests will timeout on non-vectorized code for large problems. Use JAX operations (vmap, scan, etc.) instead of Python loops.
- **Types**: Never hard-code dtypes. Use e.g. y.dtype if one needs to be specified
- **Style**: Match existing code patterns, imports, conventions
- **Fields**: Declare all dataclass fields (Equinox.Module inheritance)
- **Imports**: Problems are imported from their modules in the `__init__.py` of the 
    folder for their respective class. From there, CUTEst problems are imported in 
    `sif2jax/cutest/__init__.py`. `sif2jax/__init__.py` does not import specific 
    problems, it imports collections of problems (e.g. CUTEst).
    Each folder defines a tuple of problems - the sum of these is then `sif2jax.problems`. 
    Problems that are commented (due to requiring additional review) must be commented 
    in these tuples as well. Example: unconstrained_problems = (PROBLEM1, PROBLEM2, ...)

### 4. Testing Requirements
- **Container Required**: Tests need pycutest/Fortran libs. These are available through the bash script. **ALWAYS USE THIS SCRIPT!**. Use it with `sudo bash run_tests.sh --test-case "PROBLEM1,PROBLEM2"`, look up other handy commands in the "Quick Reference" section above.
- **Test After EVERY Change**: Even minor edits
- **Debug Order**: When fixing test failures, follow this strict priority order (matches test order in tests/test_problem.py):
  1. **Fix imports** - Ensure the problem can be imported
  2. **Fix class name** - Must match SIF problem name
  3. **Fix dimensions** - Verify n_var, n_con match SIF specifications
  4. **Fix starting point** - Check START POINT section in SIF file, must be exact
  5. **Fix objective function** - Only after starting point is correct
  6. **Fix constraint functions** - If applicable
  7. **Fix derivatives** - Gradients, Hessians, Jacobians (last priority)
  
  **Important**: Never attempt to fix functions before their inputs are correct. An incorrect starting point will make debugging the objective function impossible. Work through issues in order of increasing complexity: constants → functions → derivatives.
- **10 Attempts Rule**: After 10 failed attempts, flag for human review:
  ```python
  # TODO: Human review needed
  # Attempts made: [list attempts]
  # Suspected issues: [your analysis]
  # Resources needed: [what would help]
  ```
  If a problem is flagged for human review, comment out ALL its imports following this MANDATORY checklist:
  
  **Step 1: Comment out the problem in ALL relevant locations:**
  - In the module's `__init__.py` import statement (e.g., `sif2jax/cutest/_bounded_minimisation/__init__.py`)
  - In the module's problems tuple in the same file
  - In the main `sif2jax/cutest/__init__.py` import statement  
  - In the `cutest_problems` dictionary in `sif2jax/cutest/__init__.py`
  
  **Step 2: Run THREE verification checks (ALL must pass):**
  1. **Verify sif2jax imports successfully:**
     ```bash
     python -c "import sif2jax; print('✓ Import successful')"
     ```
  2. **Verify tests fail during collection with clear error:**
     ```bash
     sudo bash run_tests.sh tests/test_problem.py --test-case "PROBLEMNAME"
     # Should output: "RuntimeError: Test case 'PROBLEMNAME' not found in sif2jax.cutest problems"
     ```
  3. **Verify problem is NOT in the problems tuple:**
     ```python
     python -c "import sif2jax; probs = [p.__class__.__name__ for p in sif2jax.problems]; assert 'PROBLEMNAME' not in probs; print('✓ Problem successfully excluded')"
     ```
  
  **CRITICAL**: Skipping any verification step can cause expensive CI failures. All three checks MUST pass before committing.
- **Checking results of CI runs**: Numbers of problems correspond to positions (indices) in `sif2jax.problems`. 

### 5. Commit Process
- ✓ All tests pass
- ✓ Run pre-commit (NEVER use --no-verify)
- ✓ Fix all pre-commit issues
- ✓ Re-test if any changes made
- ✓ Success = clean pre-commit (no reformatting, no warnings) + all tests from bash script pass
- ✓ Commit after implementing a few problems, to keep commits compact

## Work Mode
- Once given a task, work systematically **without stopping**, in particular **DO NOT** provide summaries unless requested specifically by the user. 
- Work one problem at a time, and work your way up from passing easier tests to harder ones.
- Re-read this prompt every 20 completed items

## PR Guidelines
Run `git diff main` and summarize the diff, not just latest commits.

## Performance Target
JAX implementation must be within 5x of Fortran runtime. For larger problems JAX is expected to be faster.
**Critical**: For problems with n > 200, start with vectorized implementations. Non-vectorized code will timeout during testing.
Production-ready problems must always be vectorised.
If a sequential operation is needed, use a jax-native option such as a scan.
Common vectorization patterns:
- Replace Python for-loops with jnp operations (sum, dot, vmap)
- Use array slicing and broadcasting instead of element-wise operations
- Batch similar computations together
- For problems with repeated structure, identify the pattern and vectorize it
