from problems.knapsack import KnapsackGenerator
from problems.converter import MPSConverter, LPConverter

n_min = 11000
n_max = 20000
n_step = 1000
factor = 10 
file_name_prefix = "/Users/sebastian.senge/src/ROLEX/test_files/"
file_name_prefix = "knapsack_test_"
file_name_suffix = ".mps"

for n in range(n_min, n_max+1, n_step):
    generator = KnapsackGenerator(num_items=n, max_weight=n*factor, max_value=n*factor)
    problem = generator.generate(seed=42)
    print(f"Generated Problem: {problem}")

    file_name = file_name_prefix + str(n) + file_name_suffix
    mps_converter = MPSConverter(problem=problem)
    mps_converter.write(file_name)#"/Users/sebastian.senge/src/ROLEX/test_files/knapsack_test.mps")

    #lp_converter = LPConverter(problem=problem)
    #lp_converter.write("/Users/sebastian.senge/src/ROLEX/test_files/knapsack_test.lp")