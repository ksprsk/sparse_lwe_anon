#!/usr/bin/env python3
"""Generate plots from sparse LWE sweep CSV results."""

import os
import pandas as pd
import matplotlib.pyplot as plt

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))

SWEEPS = [
    ("m", "m"),
    ("n", "n"),
    ("k", "k"),
    ("q", "q"),
    ("sigma", "σ"),
]


def get_fixed_params(df, vary_col):
    """Extract fixed parameter info from a sweep CSV for subtitle."""
    # Common parameter columns that might appear as fixed context
    # We infer fixed params from columns that have a single unique value
    param_cols = {"m", "n", "k", "q", "sigma"}
    param_cols.discard(vary_col)
    fixed = {}
    for col in param_cols:
        if col in df.columns:
            vals = df[col].unique()
            if len(vals) == 1:
                fixed[col] = vals[0]
    return fixed


def load_sweep_data(name):
    """Load detail CSV and aggregate to get success_rate per sweep value."""
    detail_path = os.path.join(SWEEP_DIR, f"sweep_{name}_detail.csv")
    if not os.path.isfile(detail_path):
        return None
    df = pd.read_csv(detail_path)
    param_col = df.columns[0]  # first column is the swept parameter
    agg = df.groupby(param_col).agg(
        success_rate=('success', 'mean'),
        lhs=('lhs', 'first'),
        num_seeds=('seed', 'count'),
    ).reset_index()
    return agg


def plot_individual_sweep(name, xlabel):
    """Plot a single sweep CSV as success_rate vs the swept parameter."""
    df = load_sweep_data(name)
    if df is None:
        print(f"  [skip] sweep_{name}_detail.csv not found")
        return

    param_col = df.columns[0]  # first column is the swept parameter

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(df[param_col], df["success_rate"], "s-", color="tab:blue", markersize=5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Title with fixed params
    fixed = get_fixed_params(df, name)
    title = f"Sweep: vary {xlabel}"
    if fixed:
        parts = [f"{k}={v}" for k, v in sorted(fixed.items())]
        title += f"\n(fixed: {', '.join(parts)})"
    ax.set_title(title, fontsize=11)

    out_path = os.path.join(SWEEP_DIR, f"plot_sweep_{name}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [saved] {out_path}")


def plot_lhs_combined():
    """Plot all sweeps on a single LHS vs success_rate graph."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]
    found_any = False

    for i, (name, xlabel) in enumerate(SWEEPS):
        df = load_sweep_data(name)
        if df is None:
            continue
        if "lhs" not in df.columns:
            print(f"  [skip] {csv_path}: no 'lhs' column")
            continue
        ax.plot(
            df["lhs"],
            df["success_rate"],
            "s-",
            color=colors[i % len(colors)],
            markersize=5,
            label=f"vary {xlabel}",
        )
        found_any = True

    if not found_any:
        plt.close(fig)
        print("  [skip] No data for LHS combined plot")
        return

    ax.set_xlabel("LHS", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_xlim(left=0, right=0.9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.set_title("Success Rate vs LHS (all sweeps)", fontsize=12)

    out_path = os.path.join(SWEEP_DIR, "plot_lhs_combined.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [saved] {out_path}")


def main():
    print("=== Individual sweep plots ===")
    for name, xlabel in SWEEPS:
        plot_individual_sweep(name, xlabel)

    print("\n=== Combined LHS plot ===")
    plot_lhs_combined()

    print("\nDone.")


if __name__ == "__main__":
    main()
