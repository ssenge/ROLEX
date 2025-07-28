
import random
from dataclasses import dataclass, field
from typing import Dict, List

import pyomo.environ as pyo
from problems.problem import Problem

@dataclass
class ProdPlanGenerationParams:
    """Parameters for generating a Production-Distribution Planning problem."""
    num_products: int
    num_factories: int
    num_demand_centers: int
    num_periods: int
    seed: int = 42

@dataclass
class ProdPlanProblem(Problem):
    """
    Represents a Multi-Period Production-Distribution problem instance.
    This is a pure Linear Programming (LP) problem designed to be scalable and complex.
    """
    name: str
    params: ProdPlanGenerationParams
    production_costs: Dict[int, Dict[int, float]]
    inventory_costs: Dict[int, Dict[int, float]]
    transportation_costs: Dict[int, Dict[int, Dict[int, float]]]
    demand: Dict[int, Dict[int, Dict[int, float]]]
    capacity: Dict[int, Dict[int, float]]
    problem_type: str = "LP"

    def __str__(self) -> str:
        return f"{self.name} ({self.params.num_products}p, {self.params.num_factories}f, {self.params.num_demand_centers}c, {self.params.num_periods}t)"

    def to_model(self) -> pyo.ConcreteModel:
        """Converts the ProdPlan instance into a Pyomo ConcreteModel."""
        model = pyo.ConcreteModel(self.name)

        # --- Sets ---
        model.P = pyo.Set(initialize=range(self.params.num_products))
        model.F = pyo.Set(initialize=range(self.params.num_factories))
        model.C = pyo.Set(initialize=range(self.params.num_demand_centers))
        model.T = pyo.Set(initialize=range(self.params.num_periods))

        # --- Parameters ---
        model.prod_cost = pyo.Param(model.P, model.F, initialize=self.production_costs)
        model.inv_cost = pyo.Param(model.P, model.F, initialize=self.inventory_costs)
        model.trans_cost = pyo.Param(model.P, model.F, model.C, initialize=self.transportation_costs)
        model.demand = pyo.Param(model.P, model.C, model.T, initialize=self.demand)
        model.capacity = pyo.Param(model.F, model.T, initialize=self.capacity)

        # --- Variables ---
        model.produce = pyo.Var(model.P, model.F, model.T, within=pyo.NonNegativeReals)
        model.inventory = pyo.Var(model.P, model.F, model.T, within=pyo.NonNegativeReals)
        model.ship = pyo.Var(model.P, model.F, model.C, model.T, within=pyo.NonNegativeReals)

        # --- Objective Function ---
        def obj_rule(m):
            prod_cost = sum(m.prod_cost[p, f] * m.produce[p, f, t] for p in m.P for f in m.F for t in m.T)
            inv_cost = sum(m.inv_cost[p, f] * m.inventory[p, f, t] for p in m.P for f in m.F for t in m.T)
            trans_cost = sum(m.trans_cost[p, f, c] * m.ship[p, f, c, t] for p in m.P for f in m.F for c in m.C for t in m.T)
            return prod_cost + inv_cost + trans_cost
        model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # --- Constraints ---
        # 1. Inventory Balance at each factory
        def inv_balance_rule(m, p, f, t):
            # What is in inventory at the end of the last period + what is produced
            if t == 0:
                previous_inventory = 0
            else:
                previous_inventory = m.inventory[p, f, t-1]
            
            # must equal what is shipped out + what is left in inventory
            return previous_inventory + m.produce[p, f, t] == m.inventory[p, f, t] + sum(m.ship[p, f, c, t] for c in m.C)
        model.inv_balance = pyo.Constraint(model.P, model.F, model.T, rule=inv_balance_rule)

        # 2. Demand Satisfaction at each demand center
        def demand_rule(m, p, c, t):
            # Total shipments from all factories to a demand center must meet its demand
            return sum(m.ship[p, f, c, t] for f in m.F) >= m.demand[p, c, t]
        model.demand_satisfaction = pyo.Constraint(model.P, model.C, model.T, rule=demand_rule)

        # 3. Production Capacity at each factory
        def capacity_rule(m, f, t):
            # Total production of all products at a factory cannot exceed its capacity
            return sum(m.produce[p, f, t] for p in m.P) <= m.capacity[f, t]
        model.capacity_constraint = pyo.Constraint(model.F, model.T, rule=capacity_rule)

        return model

class ProdPlanGenerator:
    """Generates a ProdPlanProblem instance with random data."""
    def __init__(self, params: ProdPlanGenerationParams):
        self.params = params
        random.seed(self.params.seed)

    def generate(self) -> ProdPlanProblem:
        """Generates the random data and creates the problem instance."""
        # Generate production costs
        prod_costs = {(p, f): random.uniform(10, 50)
                      for p in range(self.params.num_products)
                      for f in range(self.params.num_factories)}

        # Generate inventory holding costs
        inv_costs = {(p, f): prod_costs[p, f] * random.uniform(0.05, 0.15)
                     for p, f in prod_costs}

        # Generate transportation costs
        trans_costs = {(p, f, c): random.uniform(1, 10)
                       for p in range(self.params.num_products)
                       for f in range(self.params.num_factories)
                       for c in range(self.params.num_demand_centers)}

        # Generate demand forecasts for each demand center
        demand = {(p, c, t): random.randint(100, 1000)
                  for p in range(self.params.num_products)
                  for c in range(self.params.num_demand_centers)
                  for t in range(self.params.num_periods)}

        # Generate factory production capacities
        # Calculate the average total demand that must be met per period.
        total_demand = sum(demand.values())
        if self.params.num_periods > 0:
            avg_demand_per_period = total_demand / self.params.num_periods
        else:
            avg_demand_per_period = 0

        # Distribute this average demand across the factories, adding a buffer.
        if self.params.num_factories > 0:
            avg_factory_capacity = (avg_demand_per_period / self.params.num_factories) * 1.5 # 50% extra capacity
        else:
            avg_factory_capacity = 0
        
        capacity = {(f, t): avg_factory_capacity
                    for f in range(self.params.num_factories)
                    for t in range(self.params.num_periods)}

        name = f"ProdPlan_{self.params.num_products}p_{self.params.num_factories}f_{self.params.num_demand_centers}c_{self.params.num_periods}t"

        return ProdPlanProblem(
            name=name,
            params=self.params,
            production_costs=prod_costs,
            inventory_costs=inv_costs,
            transportation_costs=trans_costs,
            demand=demand,
            capacity=capacity
        )
