#!/usr/bin/env python3
"""
Updated test.py using the new ROLEX client library.
"""

import client
import ommx.v1 as ommx

def create_ommx_problem():
    """Create the same optimization problem as before."""
    x1 = ommx.DecisionVariable.continuous(id=1, name="x1", lower=0, upper=10)
    x2 = ommx.DecisionVariable.continuous(id=2, name="x2", lower=0, upper=10)
    
    objective = x1 + x2
    constraint = (x1 + x2 <= 1).add_name("sum_constraint")

    instance = ommx.Instance.from_components(
        decision_variables=[x1, x2],
        objective=objective,
        constraints=[constraint],
        sense=ommx.Instance.MAXIMIZE,
    )
    
    return instance

def main():
    """Main test function using the new ROLEX client library."""
    print("🚀 ROLEX Test Client")
    print("=" * 50)
    
    # Create the optimization problem
    ommx_instance = create_ommx_problem()
    
    # Create ROLEX client and problem
    rolex_client = client.Client()
    problem = client.Problem.from_instance(ommx_instance)
    
    print(f"📋 Problem: {problem}")
    print(f"🔗 Variable mapping: {problem.get_variable_mapping()}")
    
    # Check server health
    if not rolex_client.health_check():
        print("❌ Server is not healthy. Make sure the server is running.")
        return
    
    print("✅ Server is healthy")
    
    try:
        # Submit the problem
        job_id = rolex_client.submit(problem)
        print(f"✅ Job submitted: {job_id}")
        
        # Poll for results
        result = rolex_client.poll(job_id, problem)
        
        print(f"\n📊 Result: {result}")
        
        # Show detailed analysis
        print(f"\n🔍 DETAILED ANALYSIS:")
        print(f"  Status: {result.status}")
        print(f"  Objective Value: {result.objective_value}")
        print(f"  Solver: {result.solver}")
        print(f"  Solve Time: {result.solve_time:.4f} seconds")
        print(f"  Message: {result.message}")
        
        # Variable assignments
        print(f"\n📋 Variable Assignments:")
        variables = result.get_variables()
        for var_name, var_value in variables.items():
            print(f"    {var_name} = {var_value}")
        
        
        # Objective value verification
        print(f"\n🎯 Objective Value Verification:")
        calculated_obj = result.calculate_objective_value()
        print(f"  Calculated from OMMX: {calculated_obj}")
        print(f"  Reported by server: {result.objective_value}")
        
        if abs(calculated_obj - result.objective_value) < 1e-6:
            print("  ✅ Values match!")
        else:
            print("  ⚠️  Values don't match - potential issue")
        
        # Status checks
        print(f"\n✅ Status Checks:")
        print(f"  Is optimal: {result.is_optimal()}")
        print(f"  Is feasible: {result.is_feasible()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 