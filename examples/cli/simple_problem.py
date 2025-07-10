#!/usr/bin/env python3
"""
Simple optimization problem example for ROLEX CLI.
"""

import ommx.v1 as ommx

def create_instance():
    """Create a simple optimization problem."""
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