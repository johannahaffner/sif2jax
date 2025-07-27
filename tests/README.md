# sif2jax Tests

`sif2jax` is tested against [pycutest](https://github.com/jfowkes/pycutest), a Python 
interface to the CUTEst collection of benchmark problems.

## Test Files

- `test_problem.py` - Verifies problem implementations match pycutest
- `test_runtime.py` - Benchmarks JAX vs Fortran performance
- `test_compilation.py` - Ensures JAX doesn't recompile unnecessarily (local-only)

## Quick Start

```bash
# Run all CI-safe tests
bash run_tests.sh

# Test specific problems
bash run_tests.sh --test-case "ROSENBR,ARWHEAD"

# Run memory-intensive tests locally
bash run_tests.sh --local-tests
```

## Runtime Benchmarks

Compare JAX performance against Fortran implementations:

```bash
# Run all benchmarks with output
bash run_tests.sh -k test_runtime -s

# Test specific benchmark types
bash run_tests.sh -k "test_objective_runtime" -s
bash run_tests.sh -k "test_gradient_runtime" -s

# Set custom performance threshold (default: 5.0x slower)
bash run_tests.sh --runtime-threshold=10
```

### Performance Expectations

- **Small problems (< 100 vars)**: JAX typically 1.5-3x slower (Python overhead)
- **Medium problems (100-1000 vars)**: Comparable performance
- **Large problems (> 1000 vars)**: JAX often 10-100x faster (JIT compilation)

## Configuration Options

- `--test-case`: Comma-separated list of problems to test
- `--runtime-threshold`: Maximum allowed JAX/Fortran time ratio (default: 5.0)
- `--start-at-letter`: Skip problems before specified letter
- `--local-tests`: Enable memory-intensive tests (e.g., compilation tests)

## Notes

- Compilation tests use `@pytest.mark.local_only` to prevent CI OOM errors
- Each runtime benchmark runs 3 warmups + 10 timed iterations
- JAX caches are cleared between problems for fair comparison