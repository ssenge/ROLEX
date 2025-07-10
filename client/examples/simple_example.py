#!/usr/bin/env python3
"""
Simple example of using the ROLEX client library.
"""

import sys
sys.path.insert(0, '..')
import rolex
import ommx.v1 as ommx

def create_simple_problem():
    """Create a simple optimization problem."""
    # Create decision variables
    x1 = ommx.DecisionVariable.continuous(id=1, name="x1", lower=0, upper=10)
    x2 = ommx.DecisionVariable.continuous(id=2, name="x2", lower=0, upper=10)
    
    # Create objective function: maximize x1 + x2
    objective = x1 + x2
    
    # Create constraint: x1 + x2 <= 1
    constraint = (x1 + x2 <= 1).add_name("sum_constraint")

    # Create OMMX instance
    instance = ommx.Instance.from_components(
        decision_variables=[x1, x2],
        objective=objective,
        constraints=[constraint],
        sense=ommx.Instance.MAXIMIZE,
    )
    
    return instance

def main():
    """Main example function."""
    print("🚀 ROLEX Simple Example")
    print("=" * 50)
    
    # Create the optimization problem
    ommx_instance = create_simple_problem()
    
    # Create ROLEX client and problem
    client = rolex.Client()
    problem = rolex.Problem.from_instance(ommx_instance)
    
    print(f"📋 Problem: {problem}")
    print(f"🔗 Client: {client}")
    
    # Check server health
    if not client.health_check():
        print("❌ Server is not healthy. Make sure the server is running.")
        return
    
    print("✅ Server is healthy")
    
    # Submit the problem
    try:
        job_id = client.submit(problem)
        print(f"✅ Job submitted: {job_id}")
        
        # Poll for results
        result = client.poll(job_id, problem)
        print(f"📊 Result: {result}")
        
        # Show detailed results
        if result.is_optimal():
            print("\n🎯 Detailed Results:")
            print(f"  Status: {result.status}")
            print(f"  Objective Value: {result.objective_value}")
            print(f"  Solve Time: {result.solve_time:.4f} seconds")
            print(f"  Solver: {result.solver}")
            print(f"  Variables: {result.get_variables()}")
        else:
            print(f"⚠️  Solution not optimal: {result.status}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 