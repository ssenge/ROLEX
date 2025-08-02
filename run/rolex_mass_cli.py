#!/usr/bin/env python3
"""
ROLEX Mass CLI Runner
"""
import sys
import os
import subprocess

import tempfile

def main():
    """
    Main function to generate file paths and run rolex_cli.py
    """
    if len(sys.argv) < 5:
        print("Usage: rolex_mass_cli.py <pattern> <start> <end> <step> [rolex_cli.py options]")
        print("Example: rolex_mass_cli.py test_files/knapsack_test_NUM.mps 1000 10000 1000 --solver cuopt")
        sys.exit(1)

    pattern = sys.argv[1]
    try:
        start = int(sys.argv[2])
        end = int(sys.argv[3])
        step = int(sys.argv[4])
    except ValueError:
        print("Error: start, end, and step must be integers.")
        sys.exit(1)

    rolex_cli_args = sys.argv[5:]

    files_to_process = []
    for i in range(start, end + step, step):
        files_to_process.append(pattern.replace("NUM", str(i)))

    # Find the absolute path to rolex_cli.py
    rolex_cli_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rolex_cli.py")

    if not os.path.exists(rolex_cli_path):
        print(f"Error: rolex_cli.py not found at {rolex_cli_path}")
        sys.exit(1)

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt") as tmp_batch_file:
        tmp_batch_file.write("\n".join(files_to_process))
        batch_filename = tmp_batch_file.name

    command = [sys.executable, rolex_cli_path, "--batch", batch_filename] + rolex_cli_args

    print("---")
    print(f"Executing ROLEX CLI with {len(files_to_process)} files...")
    print(f"Command: {' '.join(command)}")
    print("---")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing rolex_cli.py: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        sys.exit(1)
    finally:
        if os.path.exists(batch_filename):
            os.remove(batch_filename)


if __name__ == "__main__":
    main()
