"""
ROLEX Server - SciPy Solver
Basic fallback solver for linear programming problems using SciPy
"""
import time
import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional

from .base import BaseSolver

# Import dependencies with proper error handling
try:
    import ommx.v1 as ommx
    OMMX_AVAILABLE = True
except ImportError:
    OMMX_AVAILABLE = False
    ommx = None

try:
    from scipy.optimize import linprog
    import scipy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


logger = logging.getLogger(__name__)


class SciPyLPSolver(BaseSolver):
    """Basic linear programming solver using SciPy"""
    
    def __init__(self):
        super().__init__("SciPy-LP")
        
    def is_available(self) -> bool:
        """Check if SciPy and OMMX are available"""
        return OMMX_AVAILABLE and SCIPY_AVAILABLE
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get SciPy solver information"""
        info = {
            "name": "SciPy Linear Programming",
            "available": self.is_available(),
            "capabilities": ["linear_programming"],
            "description": "Basic LP solver using SciPy - no external dependencies required"
        }
        
        if not OMMX_AVAILABLE:
            info["ommx_status"] = "not available"
        else:
            info["ommx_status"] = "available"
            
        if not SCIPY_AVAILABLE:
            info["scipy_status"] = "not available"
            info["version"] = "unknown"
        else:
            info["scipy_status"] = "available"
            info["version"] = scipy.__version__
            
        return info
    
    def solve(self, model_dict: Dict[str, Any], parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], float, Optional[float]]:
        """
        Solve linear programming problem using SciPy
        
        Args:
            model_dict: OMMX model as dictionary (OMMX bytes format)
            parameters: Solver parameters
            
        Returns:
            Tuple of (solution_dict, solve_time, objective_value)
        """
        if not self.is_available():
            raise RuntimeError("SciPy solver not available")
        
        try:
            start_time = time.time()
            
            # Convert OMMX bytes to OMMX instance using public API
            ommx_bytes = bytes(model_dict)
            ommx_instance = ommx.Instance.from_bytes(ommx_bytes)
            
            # Convert OMMX to SciPy linprog format and solve
            result = self._solve_ommx_with_scipy(ommx_instance, parameters)
            
            solve_time = time.time() - start_time
            
            return result, solve_time, result.get("objective_value")
            
        except Exception as e:
            logger.error(f"SciPy solver failed: {str(e)}")
            raise RuntimeError(f"SciPy solver failed: {str(e)}")
    
    def _solve_ommx_with_scipy(self, ommx_instance: "ommx.Instance", parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OMMX instance to SciPy format and solve"""
        
        # Extract decision variables
        decision_vars = list(ommx_instance.decision_variables)
        n_vars = len(decision_vars)
        
        if n_vars == 0:
            raise ValueError("No decision variables found")
        
        # Build objective function coefficients
        objective_coeffs = self._extract_objective_coefficients(ommx_instance, decision_vars)
        
        # Handle maximize vs minimize
        if ommx_instance.sense == ommx.Instance.MAXIMIZE:
            # SciPy minimizes, so negate coefficients for maximization
            objective_coeffs = [-c for c in objective_coeffs]
        
        # Extract bounds
        bounds = self._extract_variable_bounds(decision_vars)
        
        # Extract constraints
        A_ub, b_ub, A_eq, b_eq = self._extract_constraints(ommx_instance, decision_vars)
        
        # Solve with SciPy
        method = parameters.get('method', 'highs')  # Use HiGHS by default (fast, open-source)
        
        logger.info(f"Solving LP with {n_vars} variables using SciPy method: {method}")
        
        result = linprog(
            c=objective_coeffs,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method=method,
            options={'presolve': True}
        )
        
        # Convert result back to OMMX format
        return self._convert_scipy_result_to_ommx(result, ommx_instance, decision_vars)
    
    def _extract_objective_coefficients(self, ommx_instance: "ommx.Instance", decision_vars: list) -> list:
        """Extract objective function coefficients"""
        n_vars = len(decision_vars)
        coeffs = [0.0] * n_vars
        
        # Map variable ID to index
        var_id_to_idx = {var.id: idx for idx, var in enumerate(decision_vars)}
        
        # Extract coefficients from objective
        objective = ommx_instance.objective
        
        # The objective is a Linear object with linear_terms and constant_term
        if hasattr(objective, 'linear_terms'):
            for var_id, coeff in objective.linear_terms.items():
                if var_id in var_id_to_idx:
                    coeffs[var_id_to_idx[var_id]] = coeff
        
        return coeffs
    
    def _extract_variable_bounds(self, decision_vars: list) -> list:
        """Extract variable bounds"""
        bounds = []
        
        for var in decision_vars:
            lower = var.lower if hasattr(var, 'lower') and var.lower is not None else 0.0
            upper = var.upper if hasattr(var, 'upper') and var.upper is not None else None
            bounds.append((lower, upper))
        
        return bounds
    
    def _extract_constraints(self, ommx_instance: "ommx.Instance", decision_vars: list) -> tuple:
        """Extract constraint matrices"""
        n_vars = len(decision_vars)
        var_id_to_idx = {var.id: idx for idx, var in enumerate(decision_vars)}
        
        A_ub = []  # Inequality constraints (<=)
        b_ub = []
        A_eq = []  # Equality constraints (=)
        b_eq = []
        
        for constraint in ommx_instance.constraints:
            # Initialize coefficient array
            coeffs = [0.0] * n_vars
            
            # Extract linear terms from the constraint function
            linear_terms = constraint.function.linear_terms
            constant_term = constraint.function.constant_term
            
            # Fill in coefficients for variables that exist in our model
            for var_id, coeff in linear_terms.items():
                if var_id in var_id_to_idx:
                    coeffs[var_id_to_idx[var_id]] = coeff
            
            # The constraint is in the form: linear_terms + constant_term <= 0
            # We need to convert this to: linear_terms <= -constant_term
            # So the RHS is -constant_term
            rhs = -constant_term
            
            # For now, treat all constraints as inequalities (<=)
            # TODO: Add support for equality constraints by checking constraint type
            A_ub.append(coeffs)
            b_ub.append(rhs)
        
        # Convert to numpy arrays if constraints exist
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        A_eq = np.array(A_eq) if A_eq else None
        b_eq = np.array(b_eq) if b_eq else None
        
        return A_ub, b_ub, A_eq, b_eq
    
    def _convert_scipy_result_to_ommx(self, result, ommx_instance: "ommx.Instance", decision_vars: list) -> Dict[str, Any]:
        """Convert SciPy result back to OMMX format"""
        
        # Map SciPy status to standard status
        status_map = {
            0: "optimal",
            1: "iteration_limit",
            2: "infeasible", 
            3: "unbounded",
            4: "numerical_failure"
        }
        
        status = status_map.get(result.status, "unknown")
        
        # Extract variable values
        variables = {}
        variable_values = {}
        
        if result.x is not None:
            for idx, var in enumerate(decision_vars):
                var_value = float(result.x[idx])
                variables[var.name] = var_value
                variable_values[var.id] = var_value
        
        # Calculate objective value (handle maximize case)
        objective_value = None
        if result.fun is not None:
            if ommx_instance.sense == ommx.Instance.MAXIMIZE:
                objective_value = -result.fun  # Negate back for maximization
            else:
                objective_value = result.fun
        
        # Create OMMX State
        ommx_state_bytes = None
        if variable_values:
            try:
                # Create OMMX State with variable assignments
                state_entries = {var_id: value for var_id, value in variable_values.items()}
                ommx_state = ommx.State(state_entries)
                ommx_state_bytes = list(ommx_state.to_bytes())
                
                logger.info(f"Created OMMX State with entries: {state_entries}")
                logger.info(f"OMMX State bytes created: {len(ommx_state_bytes)} bytes")
                
            except Exception as e:
                logger.warning(f"Failed to create OMMX State: {e}")
                ommx_state_bytes = []
        
        return {
            "status": status,
            "objective_value": objective_value,
            "variables": variables,
            "ommx_state_bytes": ommx_state_bytes or [],
            "solver_result": {
                "scipy_status": result.status,
                "scipy_message": result.message,
                "iterations": getattr(result, 'nit', 0),
                "success": result.success
            },
            "message": result.message or f"Solved with status: {status}"
        } 