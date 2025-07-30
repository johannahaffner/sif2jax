"""Create scatter plots comparing sif2jax vs pycutest runtimes.

This script creates scatter plots of the runtimes of sif2jax problems, plotted
as a function of the runtime of the Fortran implementation available through pycutest.
The runtimes should be available as a JSON file from pytest-benchmark.
The runtimes of pycutest problems should be plotted on the x axis, and the runtimes
of sif2jax problems on the y axis. The plot should be symmetric and square, and it
should have a 1:1 line on the diagonal (dashed or dotted).
The dots should be coloured according to the dimensionality of the problem. The title
is runtimes of objective functions. We need a colorbar on the right. Use a rainbow
colormap, such as jet or Spectral.

Usage:
    python scatter_runtimes.py path/to/benchmark_results.json

    # Or to use the most recent saved benchmark:
    python scatter_runtimes.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np


def load_benchmark_json(json_path):
    """Load benchmark data from pytest-benchmark JSON file.

    Args:
        json_path: Path to the JSON file containing benchmark results

    Returns:
        dict: Dictionary with arrays for sif2jax and pycutest runtimes
    """
    with open(json_path) as f:
        data = json.load(f)

    # Extract benchmark results
    benchmarks = data["benchmarks"]

    # Organize data by problem name and implementation
    problems = {}

    for bench in benchmarks:
        # Extract problem name from test name
        # (e.g., "test_sif2jax_objective_benchmark[ROSENBR]")
        test_name = bench["name"]
        problem_name = test_name.split("[")[1].rstrip("]")

        # Get implementation from group
        implementation = bench["group"].split("-")[0]  # "sif2jax" or "pycutest"

        if problem_name not in problems:
            problems[problem_name] = {}

        # Store runtime (mean in seconds) and extra info
        problems[problem_name][implementation] = {
            "runtime": bench["stats"]["mean"],
            "dimensionality": bench["extra_info"].get("dimensionality", 0),
        }

    # Filter to only problems that have both implementations
    complete_problems = {
        name: data
        for name, data in problems.items()
        if "sif2jax" in data and "pycutest" in data
    }

    # Convert to arrays
    problem_names = []
    sif2jax_runtimes = []
    pycutest_runtimes = []
    dimensionalities = []

    for name, data in sorted(complete_problems.items()):
        problem_names.append(name)
        sif2jax_runtimes.append(data["sif2jax"]["runtime"])
        pycutest_runtimes.append(data["pycutest"]["runtime"])
        dimensionalities.append(data["sif2jax"]["dimensionality"])

    # Return dictionary with numpy arrays
    return {
        "problem_names": problem_names,
        "sif2jax_runtimes": np.array(sif2jax_runtimes),
        "pycutest_runtimes": np.array(pycutest_runtimes),
        "dimensionalities": np.array(dimensionalities),
    }


def plot_runtime_scatter(data, save_path=None):
    """Create scatter plot of sif2jax vs pycutest runtimes.

    Args:
        data: Dictionary with benchmark data (from load_benchmark_json)
        save_path: Optional path to save the figure
    """
    # Extract data
    x = data["pycutest_runtimes"]
    y = data["sif2jax_runtimes"]
    sizes = data["dimensionalities"]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create scatter plot with color based on log of dimensionality
    log_sizes = np.log10(sizes + 1)  # Add 1 to avoid log(0)
    scatter = ax.scatter(
        x,
        y,
        c=log_sizes,
        s=50,
        alpha=0.7,
        cmap="jet",
        edgecolors="black",
        linewidth=0.5,
    )

    # Add colorbar with matching height
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Problem Dimensionality", fontsize=14)

    # Format colorbar ticks to show dimensions in scientific notation
    import matplotlib.ticker as ticker  # pyright: ignore[reportMissingImports]

    cbar.ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, p: f"$10^{{{int(x)}}}$" if x == int(x) else f"$10^{{{x:.1f}}}$"
        )
    )
    cbar.ax.tick_params(labelsize=12)

    # Set log scale for both axes
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Get axis limits
    min_val = min(x.min(), y.min()) * 0.5
    max_val = max(x.max(), y.max()) * 2

    # Plot 1:1 line
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        alpha=0.5,
        linewidth=1.5,
        label="1:1 line",
    )

    # Plot 5:1 line (5x slower threshold)
    ax.plot(
        [min_val, max_val],
        [min_val * 5, max_val * 5],
        "r:",
        alpha=0.5,
        linewidth=1.5,
        label="5:1 line",
    )

    # Set equal aspect ratio and limits
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")

    # Labels and title
    ax.set_xlabel("pycutest [seconds]", fontsize=14)
    ax.set_ylabel("sif2jax [seconds]", fontsize=14)
    ax.set_title("Comparison of objective function runtimes", fontsize=16)

    # Add grid
    ax.grid(True, alpha=0.3, which="both")

    # Add legend
    ax.legend(fontsize=12)

    # Set tick label sizes
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Tight layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig, ax


def find_latest_benchmark():
    """Find the most recent benchmark JSON file."""
    benchmark_dir = Path(__file__).parent.parent.parent / ".benchmarks"

    if not benchmark_dir.exists():
        return None

    # Find all JSON files recursively
    json_files = list(benchmark_dir.rglob("*.json"))

    if not json_files:
        return None

    # Sort by modification time and return the most recent
    return max(json_files, key=lambda p: p.stat().st_mtime)


def main():
    """Main function to create runtime scatter plots."""
    # Determine which JSON file to use
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        # Try to find the most recent benchmark
        json_path = find_latest_benchmark()
        if json_path is None:
            print("No benchmark JSON file specified and no saved benchmarks found.")
            print("Usage: python scatter_runtimes.py path/to/benchmark_results.json")
            return

    if not json_path.exists():
        print(f"JSON file not found at {json_path}")
        return

    # Load data
    print(f"Loading data from {json_path}")
    data = load_benchmark_json(json_path)
    print(f"Loaded {len(data['problem_names'])} problems with both implementations")

    # Create objective function plot
    output_path = Path(__file__).parent / "objective_runtime_scatter.png"
    plot_runtime_scatter(data, save_path=output_path)

    # Also show interactively if running as script
    plt.show()


if __name__ == "__main__":
    main()
