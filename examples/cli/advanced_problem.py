#!/usr/bin/env python3
"""
Advanced optimization problem example for ROLEX CLI.
"""

import ommx.v1 as ommx

def knapsack_problem(capacity=10, items=5):
    """Create a knapsack problem instance."""
    variables = []
    weights = [2, 3, 4, 5, 6][:items]
    values = [3, 4, 5, 6, 7][:items]
    
    # Create binary variables
    for i in range(items):
        var = ommx.DecisionVariable.binary(id=i+1, name=f"item_{i+1}")
        variables.append(var)
    
    # Objective: maximize total value
    objective = sum(values[i] * variables[i] for i in range(items))
    
    # Constraint: weight limit
    weight_constraint = (sum(weights[i] * variables[i] for i in range(items)) <= capacity).add_name("weight_limit")
    
    instance = ommx.Instance.from_components(
        decision_variables=variables,
        objective=objective,
        constraints=[weight_constraint],
        sense=ommx.Instance.MAXIMIZE,
    )
    
    return instance

def tsp(cities=4):
    """Create a traveling salesman problem instance."""
    # Simple TSP with distance matrix
    distances = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    
    # For simplicity, just create a basic routing problem
    variables = []
    var_id = 1
    
    for i in range(cities):
        for j in range(cities):
            if i != j:
                var = ommx.DecisionVariable.binary(id=var_id, name=f"x_{i}_{j}")
                variables.append(var)
                var_id += 1
    
    # Objective: minimize total distance
    objective = sum(distances[i][j] * variables[i*cities + j] for i in range(cities) for j in range(cities) if i != j)
    
    # Simplified constraint: each city visited once
    constraints = []
    for i in range(cities):
        constraint = (sum(variables[i*cities + j] for j in range(cities) if i != j) == 1).add_name(f"city_{i}")
        constraints.append(constraint)
    
    instance = ommx.Instance.from_components(
        decision_variables=variables,
        objective=objective,
        constraints=constraints,
        sense=ommx.Instance.MINIMIZE,
    )
    
    return instance

def production_problem(products=2, resources=3):
    """Create a production planning problem."""
    variables = []
    
    # Decision variables: production quantities
    for i in range(products):
        var = ommx.DecisionVariable.continuous(id=i+1, name=f"prod_{i+1}", lower=0, upper=100)
        variables.append(var)
    
    # Objective: maximize profit
    profits = [5, 8][:products]
    objective = sum(profits[i] * variables[i] for i in range(products))
    
    # Resource constraints
    resource_usage = [
        [2, 3],  # resource 1 usage per product
        [1, 2],  # resource 2 usage per product
        [3, 1],  # resource 3 usage per product
    ]
    
    resource_limits = [20, 15, 18]
    
    constraints = []
    for r in range(min(resources, len(resource_usage))):
        constraint = (sum(resource_usage[r][i] * variables[i] for i in range(products)) <= resource_limits[r]).add_name(f"resource_{r+1}")
        constraints.append(constraint)
    
    instance = ommx.Instance.from_components(
        decision_variables=variables,
        objective=objective,
        constraints=constraints,
        sense=ommx.Instance.MAXIMIZE,
    )
    
    return instance 