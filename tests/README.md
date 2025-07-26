# sif2jax Tests

`sif2jax` is tested against [pycutest](https://github.com/jfowkes/pycutest), a Python 
interface to the CUTEst collection of benchmark problems.

## Test Files

- `test_problem.py` - Tests problem implementations against pycutest
- `test_runtime.py` - Tests runtime performance comparisons  
- `test_compilation.py` - Tests JAX compilation behavior (local-only)

## Running Tests

### Basic Usage

```bash
# Run all tests (except local-only tests)
bash run_tests.sh

# Run tests for specific problems
bash run_tests.sh --test-case "AKIVA,ALLINITU"

# Run tests starting from a specific letter
bash run_tests.sh --start-at-letter M
```

### Local-Only Tests

Some tests are marked as "local-only" because they may use excessive memory or are not suitable for CI environments. These tests are skipped by default.

To run local-only tests (including compilation tests):

```bash
# Run all tests including local-only
bash run_tests.sh --local-tests

# Run only compilation tests locally
bash run_tests.sh --local-tests -k test_compilation
```

The `@local_only` decorator is used to mark tests that should only run when the `--local-tests` flag is passed. This is particularly useful for:

- Compilation tests that verify JAX doesn't recompile functions unnecessarily
- Tests that use significant memory and might cause OOM errors in CI
- Performance benchmarks that need consistent hardware

## Test Configuration

The test suite uses several pytest options configured in `conftest.py`:

- `--test-case`: Run only specified test cases (comma-separated)
- `--runtime-threshold`: Set threshold for runtime comparison tests
- `--start-at-letter`: Skip problems before the specified letter
- `--local-tests`: Enable local-only tests

## Memory Management

The compilation tests include an autouse fixture that clears JAX and Equinox caches after each test to prevent memory buildup. This is only active in the compilation test module to avoid slowing down other tests.

## TODO

- Tests for documentation: input, classification number, source, dates