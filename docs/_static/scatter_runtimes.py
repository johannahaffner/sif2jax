"""Create scatter plots comparing sif2jax vs pycutest runtimes.

This script creates scatter plots of the runtimes of sif2jax problems, plotted
as a function of the runtime of the Fortran implementation available through pycutest.
The runtimes should be available as a .csv file, stored in the benchmarks folder.
The runtimes of pycutest problems should be plotted on the x axis, and the runtimes
of sif2jax problems on the y axis. The plot should be symmetric and square, and it
should have a 1:1 line on the diagonal (dashed or dotted).
The dots should be coloured according to the dimensionality of the problem. The title
is runtimes of objective functions. We need a colorbar on the right. Use a rainbow
colormap, such as jet or Spectral.

NOTE: This script assumes benchmark data from fully tested problems.
It should be run after benchmarks/runtimes.py has successfully completed
on a branch where all tests pass.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np


def load_benchmark_data(csv_path):
    """Load benchmark data from CSV file.

    Args:
        csv_path: Path to the CSV file containing benchmark results

    Returns:
        dict: Dictionary with arrays for each column
    """
    data = {
        "problem_name": [],
        "dimensionality": [],
        "sif2jax_objective_runtime": [],
        "pycutest_objective_runtime": [],
        "sif2jax_constraint_runtime": [],
        "pycutest_constraint_runtime": [],
    }

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["problem_name"].append(row["problem_name"])
            data["dimensionality"].append(int(row["dimensionality"]))
            data["sif2jax_objective_runtime"].append(
                float(row["sif2jax_objective_runtime"])
            )
            data["pycutest_objective_runtime"].append(
                float(row["pycutest_objective_runtime"])
            )

            # Handle NaN values for constraints
            sif2jax_cons = row["sif2jax_constraint_runtime"]
            pycutest_cons = row["pycutest_constraint_runtime"]
            data["sif2jax_constraint_runtime"].append(
                float(sif2jax_cons) if sif2jax_cons != "nan" else np.nan
            )
            data["pycutest_constraint_runtime"].append(
                float(pycutest_cons) if pycutest_cons != "nan" else np.nan
            )

    # Convert to numpy arrays
    for key in data:
        if key != "problem_name":
            data[key] = np.array(data[key])  # pyright: ignore[reportGeneralTypeIssues]

    return data


def plot_runtime_scatter(data, save_path=None):
    """Create scatter plot of sif2jax vs pycutest runtimes.

    Args:
        data: Dictionary with benchmark data (from load_benchmark_data)
        save_path: Optional path to save the figure
    """
    # Extract data
    x = data["pycutest_objective_runtime"]
    y = data["sif2jax_objective_runtime"]
    sizes = data["dimensionality"]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create scatter plot with color based on dimensionality
    scatter = ax.scatter(
        x, y, c=sizes, s=50, alpha=0.7, cmap="jet", edgecolors="black", linewidth=0.5
    )

    # Add colorbar
    plt.colorbar(scatter, ax=ax, label="Problem Dimensionality")

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

    # Set equal aspect ratio and limits
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")

    # Labels and title
    ax.set_xlabel("pycutest Runtime (seconds)", fontsize=12)
    ax.set_ylabel("sif2jax Runtime (seconds)", fontsize=12)
    ax.set_title(
        "Runtime Comparison: sif2jax vs pycutest Objective Functions", fontsize=14
    )

    # Add grid
    ax.grid(True, alpha=0.3, which="both")

    # Add legend
    ax.legend()

    # Tight layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_constraint_runtime_scatter(data, save_path=None):
    """Create scatter plot of sif2jax vs pycutest constraint runtimes.

    Args:
        data: Dictionary with benchmark data (from load_benchmark_data)
        save_path: Optional path to save the figure
    """
    # Filter out problems without constraints (NaN values)
    mask = ~np.isnan(data["sif2jax_constraint_runtime"])

    if not np.any(mask):
        print("No problems with constraints found in the data.")
        return None, None

    # Extract data for problems with constraints
    x = data["pycutest_constraint_runtime"][mask]
    y = data["sif2jax_constraint_runtime"][mask]
    sizes = data["dimensionality"][mask]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create scatter plot with color based on dimensionality
    scatter = ax.scatter(
        x, y, c=sizes, s=50, alpha=0.7, cmap="jet", edgecolors="black", linewidth=0.5
    )

    # Add colorbar
    plt.colorbar(scatter, ax=ax, label="Problem Dimensionality")

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

    # Set equal aspect ratio and limits
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")

    # Labels and title
    ax.set_xlabel("pycutest Runtime (seconds)", fontsize=12)
    ax.set_ylabel("sif2jax Runtime (seconds)", fontsize=12)
    ax.set_title(
        "Runtime Comparison: sif2jax vs pycutest Constraint Functions", fontsize=14
    )

    # Add grid
    ax.grid(True, alpha=0.3, which="both")

    # Add legend
    ax.legend()

    # Tight layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def main():
    """Main function to create runtime scatter plots."""
    # Path to the CSV file
    csv_path = (
        Path(__file__).parent.parent.parent / "benchmarks" / "runtimes_latest.csv"
    )

    if not csv_path.exists():
        print(f"CSV file not found at {csv_path}")
        return

    # Load data
    print(f"Loading data from {csv_path}")
    data = load_benchmark_data(csv_path)
    print(f"Loaded {len(data['problem_name'])} problems")

    # Create objective function plot
    plot_runtime_scatter(data, save_path="objective_runtime_scatter.png")
    plt.show()

    # Create constraint function plot
    fig2, _ = plot_constraint_runtime_scatter(
        data, save_path="constraint_runtime_scatter.png"
    )
    if fig2:
        plt.show()


if __name__ == "__main__":
    main()
