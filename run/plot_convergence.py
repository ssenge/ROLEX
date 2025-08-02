#!/usr/bin/env python3
"""
Plot convergence data for two specified problems from a results.csv file.
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
        description="Plot convergence data for two specified problems from a results.csv file.",
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
        help="Path to save the output plot image (e.g., convergence.png)."
    )
    parser.add_argument(
        "--problems",
        type=str,
        nargs=2,
        required=True,
        help="Two problem filenames to compare."
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Convergence Plot",
        help="Custom title for the plot."
    )

    args = parser.parse_args()

    # --- Data Processing with Pandas ---
    try:
        df = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        return

    problem1_data = df[df['input_filename'] == args.problems[0]]
    problem2_data = df[df['input_filename'] == args.problems[1]]

    if problem1_data.empty:
        print(f"Error: Problem '{args.problems[0]}' not found in the input file.")
        return
    if problem2_data.empty:
        print(f"Error: Problem '{args.problems[1]}' not found in the input file.")
        return

    # --- Parse Convergence Data ---
    def parse_convergence(data_str):
        times = []
        objectives = []
        if pd.isna(data_str):
            return times, objectives
        points = data_str.split(';')
        for point in points:
            if ':' in point:
                try:
                    time, objective = point.split(':')
                    times.append(float(time))
                    objectives.append(float(objective))
                except ValueError:
                    # Skip malformed points
                    continue
        return times, objectives

    times1, objectives1 = parse_convergence(problem1_data.iloc[0]['convergence_objectives'])
    times2, objectives2 = parse_convergence(problem2_data.iloc[0]['convergence_objectives'])

    # --- Plotting with Matplotlib/Seaborn ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    plt.plot(times1, objectives1, marker='o', linestyle='-', label=args.problems[0])
    plt.plot(times2, objectives2, marker='o', linestyle='-', label=args.problems[1])

    # --- Final Touches & Output ---
    plt.title(args.title, fontsize=16)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Objective Value", fontsize=12)
    plt.legend(title='Problem')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()

    try:
        plt.savefig(args.output_file, dpi=300)
        print(f"Plot successfully saved to {args.output_file}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    main()
