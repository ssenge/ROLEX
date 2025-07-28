
import os
import sys

from problems.prodplan import ProdPlanGenerator, ProdPlanGenerationParams
from problems.converter import MPSConverter

# --- Configuration ---
# Fixed Parameters
SEED = 42
OUTPUT_DIR = "/Users/sebastian.senge/src/ROLEX/test_files"

# Loop Ranges
products_min, products_max, products_step = 100, 101, 10
factories_min, factories_max, factories_step = 30, 31, 1
demand_centers_min, demand_centers_max, demand_centers_step = 14, 21, 1
periods_min, periods_max, periods_step = 50, 51, 1

def main():
    """
    Generates multiple ProdPlan problem instances based on specified ranges
    for products, factories, demand centers, and periods.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for num_products in range(products_min, products_max, products_step):
        for num_factories in range(factories_min, factories_max, factories_step):
            for num_demand_centers in range(demand_centers_min, demand_centers_max, demand_centers_step):
                for num_periods in range(periods_min, periods_max, periods_step):
                    print(f"--- Generating ProdPlan with {num_products}p, {num_factories}f, {num_demand_centers}c, {num_periods}t ---")

                    gen_params = ProdPlanGenerationParams(
                        num_products=num_products,
                        num_factories=num_factories,
                        num_demand_centers=num_demand_centers,
                        num_periods=num_periods,
                        seed=SEED
                    )

                    try:
                        generator = ProdPlanGenerator(params=gen_params)
                        problem = generator.generate()
                        print(f"Generated Problem: {problem}")

                        file_name = f"{problem.name}.mps"
                        output_path = os.path.join(OUTPUT_DIR, file_name)
                        
                        mps_converter = MPSConverter(problem=problem)
                        mps_converter.write(output_path)
                        print(f"Successfully wrote to {output_path}")

                    except Exception as e:
                        print(f"Error generating or writing problem: {e}")

if __name__ == "__main__":
    main()
