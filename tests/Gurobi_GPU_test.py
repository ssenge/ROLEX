import gurobipy as gp
import sys
import io

def run_gurobi_gpu_test():
    print("Starting Gurobi GPU test (MILP)...")

    # Capture Gurobi output
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    try:
        # Create a new model
        m = gp.Model("gurobi_gpu_milp_test")

        # Set parameters to encourage GPU usage (Barrier method)
        m.setParam('OutputFlag', 1) # Ensure output is printed
        m.setParam('LogToConsole', 1) # Ensure log goes to console (and thus our StringIO)
        m.setParam('Method', 2) # Barrier method, often GPU-accelerated for LPs
        m.setParam('Presolve', 2) # Aggressive presolve
        # For MILP, Gurobi might use different strategies, but Barrier is relevant for the LP relaxation.

        # Add variables
        x = m.addVar(vtype=gp.GRB.CONTINUOUS, name="x")
        y = m.addVar(vtype=gp.GRB.INTEGER, name="y") # Make y an integer variable
        z = m.addVar(vtype=gp.GRB.BINARY, name="z") # Make z a binary variable

        # Set objective: Maximize 3x + 2y + 5z
        m.setObjective(3*x + 2*y + 5*z, gp.GRB.MAXIMIZE)

        # Add constraints
        m.addConstr(x + y + z <= 10, "c0")
        m.addConstr(2*x - y >= 2, "c1")
        m.addConstr(y + 3*z <= 8, "c2")
        m.addConstr(x >= 0, "x_bound")
        m.addConstr(y >= 0, "y_bound")

        # Optimize the model
        m.optimize()

        # Check optimization status
        if m.status == gp.GRB.OPTIMAL:
            print(f"Optimization successful. Objective value: {m.objVal}")
            print(f"x = {x.X}, y = {y.X}, z = {z.X}")
        else:
            print(f"Optimization failed with status: {m.status}")

    except gp.GurobiError as e:
        print(f'Error code {e.errno}: {e}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
    finally:
        # Restore stdout
        sys.stdout = old_stdout

    # Analyze the captured output for GPU indicators
    gurobi_log = redirected_output.getvalue()
    print("\n--- Gurobi Log Output ---")
    print(gurobi_log)
    print("-------------------------")

    gpu_detected = False
    gpu_keywords = ["GPU", "cuOpt", "CUDA", "NVIDIA", "accelerated", "device"] # Added 'device' as a general keyword

    for keyword in gpu_keywords:
        if keyword.lower() in gurobi_log.lower():
            gpu_detected = True
            break

    if gpu_detected:
        print("\n✅ GPU usage detected in Gurobi log.")
    else:
        print("\n❌ No explicit GPU usage detected in Gurobi log. This might mean:")
        print("   - Gurobi did not utilize the GPU for this problem.")
        print("   - The problem was too small or simple to warrant GPU acceleration.")
        print("   - The GPU/CUDA setup is not correctly configured or detected by Gurobi.")
        print("   - The Gurobi version does not support cuOpt or GPU acceleration for this problem type.")
        print("   - The log output does not contain the expected keywords.")

if __name__ == "__main__":
    run_gurobi_gpu_test()
