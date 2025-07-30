#!/usr/bin/env python3
"""Benchmark sif2jax problems against pycutest implementations."""

import csv
import timeit
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pycutest  # pyright: ignore[reportMissingImports]  - runs in container
import sif2jax


# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


def benchmark_pycutest(
    func: Callable, *args, number: int = 100, repeat: int = 5
) -> float:
    """Benchmark a pycutest function using timeit.

    Args:
        func: Function to benchmark
        *args: Arguments to pass to func
        number: Number of executions per timing (default: 100)
        repeat: Number of timing repetitions (default: 5)

    Returns:
        Best average time per execution in seconds
    """
    _ = func(*args)  # Warm up
    timer = timeit.Timer(lambda: func(*args))
    times = timer.repeat(repeat=repeat, number=number)
    return min(times) / number


def benchmark_jax(
    func: Callable, args: tuple, number: int = 100, repeat: int = 5
) -> float:
    """Benchmark a JAX function using AOT compilation and timeit.

    Args:
        func: JAX function to benchmark (should already be jitted)
        args: Arguments to pass to func as a tuple
        number: Number of executions per timing (default: 100)
        repeat: Number of timing repetitions (default: 5)

    Returns:
        Best average time per execution in seconds
    """
    compiled = func.lower(*args).compile()  # AOT compile
    _ = jtu.tree_map(lambda x: x.block_until_ready(), compiled(*args))  # Warm up

    # TODO: tree map has a minor overhead, only needed for output constraint tuple
    # Perhaps rewrite this function so that it is not needed - this means writing a
    # separate function for the constraint method output.
    def timed_call():
        out = compiled(*args)
        return jtu.tree_map(lambda x: x.block_until_ready(), out)

    timer = timeit.Timer(timed_call)
    times = timer.repeat(repeat=repeat, number=number)
    return min(times) / number


def benchmark_problem(problem):
    """Benchmark a single problem and return timing results.

    Args:
        problem: sif2jax problem instance

    Returns:
        OrderedDict with benchmark results
    """
    result = OrderedDict()
    result["problem_name"] = problem.name
    result["dimensionality"] = problem.y0.size

    pycutest_problem = pycutest.import_problem(problem.name, drop_fixed_variables=False)
    pycutest_obj_time = benchmark_pycutest(pycutest_problem.obj, problem.y0)
    jax_obj_time = benchmark_jax(jax.jit(problem.objective), (problem.y0, problem.args))

    result["sif2jax_objective_runtime"] = jax_obj_time
    result["pycutest_objective_runtime"] = pycutest_obj_time

    # Benchmark constraint function if it exists
    if hasattr(problem, "constraint"):
        pycutest_cons_time = benchmark_pycutest(pycutest_problem.cons, problem.y0)
        jax_cons_time = benchmark_jax(jax.jit(problem.constraint), (problem.y0,))

        result["sif2jax_constraint_runtime"] = jax_cons_time
        result["pycutest_constraint_runtime"] = pycutest_cons_time
    else:
        result["sif2jax_constraint_runtime"] = jnp.nan
        result["pycutest_constraint_runtime"] = jnp.nan

    jax.clear_caches()  # Fair comparison
    eqx.clear_caches()  # Clear Equinox cache too (should be redundant here)
    pycutest.clear_cache(problem.name)  # Clear pycutest cache
    return result


def write_csv(results, filename):
    """Write benchmark results to CSV file.

    Args:
        results: List of OrderedDict with benchmark results
        filename: Output CSV filename
    """
    if not results:
        print("No results to write.")
        return

    # Get headers from the first result
    headers = list(results[0].keys())

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)


def main():
    """Run benchmarks on all sif2jax problems.

    NOTE: This script assumes all problems have been fully tested and verified.
    It should only be run on branches where all tests pass, as it assumes:
    - All problems have corresponding pycutest implementations
    - All objective/constraint functions run without errors
    - Runtimes are within acceptable bounds (5x of Fortran)

    For partial or experimental branches, manually select specific problems
    or add error handling as needed.
    """
    results = []
    for i, problem in enumerate(sif2jax.problems):
        print(f"Benchmarking problem {i + 1}/{len(sif2jax.problems)}: {problem.name}")
        result = benchmark_problem(problem)
        results.append(result)

    # Write results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmarks/runtimes_{timestamp}.csv"
    write_csv(results, output_file)
    print(f"\nResults written to {output_file}")

    # Also write a latest version for easy access
    write_csv(results, "benchmarks/runtimes_latest.csv")
    print("Results also written to benchmarks/runtimes_latest.csv")


if __name__ == "__main__":
    main()
