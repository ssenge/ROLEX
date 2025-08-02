#!/usr/bin/env python3
"""
Plot performance results from a results.csv file.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """
    Main function to generate the plot.
    """
    parser = argparse.ArgumentParser(
        description="Plot performance results from a results.csv file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input results.csv file."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save the output plot image (e.g., performance.png)."
    )
    parser.add_argument(
        "--y-log-scale",
        action="store_true",
        help="Use a logarithmic scale for the y-axis (Time to Solution)."
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Solver Performance: Time to Solution vs. Number of Variables",
        help="Custom title for the plot."
    )

    args = parser.parse_args()

    # --- Data Processing with Pandas ---
    try:
        # Explicitly set index_col=None to ensure input_filename is a regular column
        df = pd.read_csv(args.input_file, index_col=None)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        return

    # --- Correct Column Mapping and Data Cleaning ---
    # Columns are now correctly named in the CSV
    df['solver_engine'] = df['solver_engine']
    df['num_variables'] = pd.to_numeric(df['num_variables'], errors='coerce')
    df['time_to_solution'] = pd.to_numeric(df['time_to_solution'], errors='coerce')

    # Drop rows where key data is missing or invalid after remapping
    df.dropna(subset=['num_variables', 'time_to_solution', 'solver_engine'], inplace=True)

    # --- Plotting with Matplotlib/Seaborn ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    solvers = df['solver_engine'].unique()
    colors = sns.color_palette("husl", len(solvers))

    # Group data and calculate statistics for each solver
    for i, solver in enumerate(solvers):
        solver_data = df[df['solver_engine'] == solver]
        
        # Aggregate stats for this solver
        agg_stats = solver_data.groupby('num_variables')['time_to_solution'].agg([
            'mean', 
            'std'
        ]).reset_index()
        agg_stats['std'].fillna(0, inplace=True)

        plt.errorbar(
            x=agg_stats['num_variables'],
            y=agg_stats['mean'],
            yerr=agg_stats['std'],
            label=solver,
            fmt='-o',  # Format: line with markers
            capsize=5, # Error bar cap size
            color=colors[i]
        )

    # --- Final Touches & Output ---
    plt.title(args.title, fontsize=16)
    plt.xlabel('Number of Variables', fontsize=12)
    plt.ylabel('Average Time to Solution (seconds)', fontsize=12)
    
    # Disable scientific notation on x-axis
    plt.ticklabel_format(style='plain', axis='x')
    
    if args.y_log_scale:
        plt.yscale('log')
        plt.ylabel('Average Time to Solution (seconds) - Log Scale', fontsize=12)

    plt.legend(title='Solver Engine')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()

    try:
        plt.savefig(args.output_file, dpi=300)
        print(f"Plot successfully saved to {args.output_file}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    main()