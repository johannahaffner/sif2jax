# Runtime Benchmarking

The `test_runtime.py` file provides runtime benchmarks comparing our JAX implementations against pycutest.

## Usage

### Basic usage with default problems:
```bash
bash run_tests.sh tests/test_runtime.py -s
```

### Test specific problems:
```bash
bash run_tests.sh tests/test_runtime.py -s --test-case="ROSENBR,ARWHEAD,HS10"
```

### Set custom runtime threshold (default is 5.0):
```bash
bash run_tests.sh tests/test_runtime.py -s --runtime-threshold=10
```

### Run only specific benchmark types:
```bash
# Only objective function benchmarks
bash run_tests.sh tests/test_runtime.py -s -k "test_objective_runtime"

# Only gradient benchmarks  
bash run_tests.sh tests/test_runtime.py -s -k "test_gradient_runtime"

# Only combined obj+grad benchmarks
bash run_tests.sh tests/test_runtime.py -s -k "test_combined_runtime"
```

## Interpreting Results

The benchmark reports show:
- **pycutest**: Time taken by the original Fortran implementation (in milliseconds)
- **JAX**: Time taken by our JAX implementation (in milliseconds)
- **Ratio**: JAX time / pycutest time

A ratio < 1.0 means JAX is faster, > 1.0 means pycutest is faster.

### Expected Performance Patterns

1. **Small problems (< 100 variables)**: JAX is typically 1.5-3x slower due to Python/JAX overhead
2. **Medium problems (100-1000 variables)**: Performance is comparable
3. **Large problems (> 1000 variables)**: JAX can be 10-100x faster due to JIT compilation and vectorization

## Implementation Details

- Each function is benchmarked with 3 warmup runs followed by 10 timed iterations
- JAX cache is cleared between problems to ensure fair comparison
- Constrained problems skip the combined obj+grad test (different pycutest API)
- Tests fail if the ratio exceeds the threshold (default: 5.0)

## Adding New Benchmarks

To add new types of benchmarks, extend the `TestRuntime` class in `test_runtime.py` with new test methods following the pattern of existing tests.