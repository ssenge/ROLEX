#!/usr/bin/env python3
"""
Generate VRP problem instances with random graphs and depots.
"""

import os
import sys

from problems.vrp import VRPGenerator, VRPGenerationParams
from problems.converter import MPSConverter

# Fixed Parameters
MIN_DEMAND = 5
MAX_DEMAND = 25
AVG_DEGREE = 3.0
CAPACITY_MULTIPLIER = 1.5
SEED = 42
OUTPUT_DIR = "/Users/sebastian.senge/src/ROLEX/test_files"

# Loop Ranges
nodes_min, nodes_max, nodes_step = 90, 110, 10
vehicles_min, vehicles_max, vehicles_step = 2, 7, 1
depots_min, depots_max, depots_step = 2, 4, 1

def main():
    """
    Generates multiple VRP problem instances based on specified ranges
    for nodes, vehicles, and depots.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for num_nodes in range(nodes_min, nodes_max, nodes_step):
        for num_vehicles in range(vehicles_min, vehicles_max, vehicles_step):
            for num_depots in range(depots_min, depots_max, depots_step):
                # Ensure num_depots is not greater than num_nodes
                if num_depots >= num_nodes:
                    continue

                gen_params = VRPGenerationParams(
                    num_nodes=num_nodes,
                    num_vehicles=num_vehicles,
                    num_depots=num_depots,
                    min_demand=MIN_DEMAND,
                    max_demand=MAX_DEMAND,
                    avg_degree=AVG_DEGREE,
                    capacity_multiplier=CAPACITY_MULTIPLIER,
                    seed=SEED
                )

                try:
                    generator = VRPGenerator(params=gen_params)
                    problem = generator.generate()
                    print(f"Generated Problem: {problem}")

                    file_name = f"{problem.name}.mps"
                    output_path = os.path.join(OUTPUT_DIR, file_name)
                    
                    mps_converter = MPSConverter(problem=problem)
                    mps_converter.write(output_path)
                    print(f"Successfully wrote to {output_path}")

                except (ValueError, RuntimeError) as e:
                    print(f"Skipping combination ({num_nodes}n, {num_vehicles}v, {num_depots}d): {e}")


if __name__ == "__main__":
    main()
