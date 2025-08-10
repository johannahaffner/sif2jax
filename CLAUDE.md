# SIF2JAX Conversion Assistant

## Mission
Convert CUTEST problems to JAX implementations that match Fortran precision and are as performant as possible.

## Quick Reference

```bash
sudo bash run_tests.sh tests/test_problem.py --test-case "PROBLEM1,PROBLEM2"   # Test specific problems
sudo bash run_tests.sh tests/test_problem.py -k test_objective                 # Test specific aspect
sudo bash run_tests.sh --test-case "PROBLEM1" --local-tests  # Additionally test compilation
ruff format . && ruff check .                       # Format and lint
```

(The test script may have to be run with `sudo bash` in the container.)

## Workflow: Implement → Test → Fix → Commit → Repeat

### 1. Overall goal
Problems only count as implemented if they pass the tests against pycutest in the main test suite, accessible through the bash script. 
This means that `sudo bash run_tests.sh --test-case "PROBLEM1" --local-tests` should pass.
The tests are designed to be very informative, and can guide you toward a working implementation.

### 2. Implementation Priority
SIF problems have a group-separable structure. Identifying this structure helps to identify opportunities for vectorisation and batched operations, as well as to `divide and conquer` complex problems.
You can look up some guiding principles in the CONVERSION_GUIDE file.

**Background on partially separable functions:** Griewank, A., Toint, Ph.L.: "On the unconstrained optimization of partially separable functions" in Powell, M.J.D. (ed.) Nonlinear Optimization 1981, pp. 301–312, Academic Press, London (1982). This foundational paper introduces the concept of functions that can be expressed as sums of element functions, enabling efficient derivative computation through the chain rule on separable components.

Here are the sources relevant to problem implementations:
1. **SIF Files**: `archive/mastsif/` folder - Original SIF problem definitions (PRIMARY SOURCE). Look in this folder whenever you are asked to find problems of any series - e.g. search for "TRO" in this folder to find problems from the "TRO" series of problems. You can also consult the missing_problems.md file, but this won't always be 100 % up to date.
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
- **Batch Testing**: Full test suite will result in timeout in devcontainer, test problems individually or in small batches instead. 
- **Test Timeouts**: If a problem is poorly vectorised, its tests may time out. In this case, vectorise the problem before trying again.
- **10 Attempts Rule**: After 10 failed attempts, flag for human review:
  ```python
  # TODO: Human review needed
  # Attempts made: [list attempts]
  # Suspected issues: [your analysis]
  # Resources needed: [what would help]
  ```
  If a problem is flagged for human review, its imports should be commented out. 
  Verify that it cannot be run anymore by trying to run the tests for it, these should then fail during collection with a clear error message. 
  When commenting out a problem that is marked for human review, also try running `python -c import sif2jax` to confirm that the removal still permits importing the library as a whole and does not result in errata.
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
You can use for-loops to write an initial implementation that passes the tests, but please convert to a vectorised implementation once you have achieved this step.
Production-ready problems should always be vectorised.
If a sequential operation is needed, use a jax-native option such as a scan.