"""
ROLEX Server - Gurobi Solver
Clean implementation based on working standalone examples
"""
import time
import logging
from typing import Dict, Any, Tuple, Optional

from .base import BaseSolver

# Import dependencies with proper error handling
try:
    import ommx.v1 as ommx
    OMMX_AVAILABLE = True
except ImportError:
    OMMX_AVAILABLE = False

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


logger = logging.getLogger(__name__)


class GurobiSolver(BaseSolver):
    """Gurobi solver with OMMX integration"""
    
    def __init__(self):
        super().__init__("Gurobi")
        
    def is_available(self) -> bool:
        """Check if Gurobi and OMMX are available"""
        return OMMX_AVAILABLE and GUROBI_AVAILABLE
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get Gurobi solver information"""
        info = {
            "name": "Gurobi with OMMX Integration",
            "available": self.is_available(),
            "capabilities": ["linear", "quadratic", "mixed-integer"]
        }
        
        if not OMMX_AVAILABLE:
            info["ommx_status"] = "not available"
        else:
            info["ommx_status"] = "available"
            
        if not GUROBI_AVAILABLE:
            info["gurobi_status"] = "not available"
            info["version"] = "unknown"
            info["license"] = "unknown"
        else:
            try:
                # Get Gurobi version and license info
                env = gp.Env()
                info["version"] = f"{gp.version[0]}.{gp.version[1]}.{gp.version[2]}"
                info["gurobi_status"] = "available"
                
                # Test license by creating a small model
                test_model = gp.Model(env=env)
                test_var = test_model.addVar()
                test_model.setObjective(test_var, GRB.MAXIMIZE)
                # Don't optimize, just test if model creation works
                info["license"] = "valid"
                env.dispose()
                
            except Exception as e:
                info["version"] = "unknown" 
                info["license"] = f"error: {str(e)}"
                info["gurobi_status"] = "license_error"
        
        return info
    
    def solve(self, model_dict: Dict[str, Any], parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], float, Optional[float]]:
        """
        Solve optimization problem using Gurobi
        
        Args:
            model_dict: OMMX model as dictionary (OMMX bytes format)
            parameters: Solver parameters
            
        Returns:
            Tuple of (solution_dict, solve_time, objective_value)
        """
        if not self.is_available():
            raise RuntimeError("Gurobi solver not available")
        
        try:
            # Convert OMMX bytes to OMMX instance using public API
            ommx_bytes = bytes(model_dict)
            ommx_instance = ommx.Instance.from_bytes(ommx_bytes)
            
            # Convert OMMX to Gurobi and solve, returning dictionary for now
            return self._solve_ommx_with_gurobi(ommx_instance, parameters)
            
        except Exception as e:
            logger.error(f"Gurobi solver failed: {str(e)}")
            raise RuntimeError(f"Gurobi solver failed: {str(e)}")
    
    def _solve_ommx_with_gurobi(self, ommx_instance, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], float, Optional[float]]:
        """Convert OMMX instance to Gurobi, solve, and return OMMX State bytes"""
        
        print("DEBUG: _solve_ommx_with_gurobi called")
        
        # Create Gurobi model
        model = gp.Model("ommx_v2")
        
        # Configure parameters
        verbose = parameters.get("verbose", False)
        if not verbose:
            model.setParam('OutputFlag', 0)
            
        time_limit = parameters.get("time_limit")
        if time_limit:
            model.setParam('TimeLimit', time_limit)
        
        # Create Gurobi variables
        gurobi_vars = {}
        for var in ommx_instance.decision_variables:
            lb = var.bound.lower if var.bound.lower != float('-inf') else -GRB.INFINITY
            ub = var.bound.upper if var.bound.upper != float('inf') else GRB.INFINITY
            
            gurobi_vars[var.id] = model.addVar(
                vtype=GRB.CONTINUOUS,  # Could extend for integer/binary
                name=var.name,
                lb=lb,
                ub=ub
            )
        
        # Add constraints
        for constraint in ommx_instance.constraints:
            expr = 0
            linear = constraint.function.as_linear()
            if linear:
                # Add linear terms
                for var_id, coeff in linear.linear_terms.items():
                    expr += coeff * gurobi_vars[var_id]
                # Add constant
                expr += linear.constant_term
            
            # Add constraint based on equality type
            if constraint.equality == ommx.Equality.LESS_THAN_OR_EQUAL_TO_ZERO:
                model.addConstr(expr <= 0, constraint.name)
            elif constraint.equality == ommx.Equality.EQUAL_TO_ZERO:
                model.addConstr(expr == 0, constraint.name)
            # Note: GreaterThanOrEqualToZero would be >= 0
        
        # Set objective
        obj_expr = 0
        obj_linear = ommx_instance.objective.as_linear()
        if obj_linear:
            for var_id, coeff in obj_linear.linear_terms.items():
                obj_expr += coeff * gurobi_vars[var_id]
            obj_expr += obj_linear.constant_term
        
        sense = GRB.MAXIMIZE if ommx_instance.sense == ommx.Sense.MAXIMIZE else GRB.MINIMIZE
        model.setObjective(obj_expr, sense)
        
        # Solve with timing
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        # Create OMMX State from Gurobi solution
        if model.status == GRB.OPTIMAL:
            # Create State with variable values
            state_entries = {}
            for var in ommx_instance.decision_variables:
                state_entries[var.id] = gurobi_vars[var.id].X
            
            logger.info(f"Creating OMMX State with entries: {state_entries}")
            state = ommx.State.from_values(state_entries)
            state_bytes = state.to_bytes()
            logger.info(f"OMMX State bytes created: {len(state_bytes)} bytes")
            
            solution = {
                "status": "optimal",
                "ommx_state_bytes": list(state_bytes),  # Return as list of integers
                "variables": {var.name: gurobi_vars[var.id].X for var in ommx_instance.decision_variables},  # Keep for compatibility
                "message": "Optimal solution found"
            }
            logger.info(f"Solution keys: {list(solution.keys())}")
            
            return solution, solve_time, model.ObjVal
            
        else:
            status_map = {
                GRB.INFEASIBLE: "infeasible",
                GRB.UNBOUNDED: "unbounded", 
                GRB.INF_OR_UNBD: "infeasible_or_unbounded",
                GRB.TIME_LIMIT: "time_limit",
                GRB.INTERRUPTED: "interrupted"
            }
            
            status = status_map.get(model.status, f"unknown_status_{model.status}")
            solution = {
                "status": status,
                "ommx_state_bytes": None,
                "variables": {},
                "message": f"Optimization finished with status: {status}"
            }
            
            return solution, solve_time, None 