"""
ROLEX Server - Base Solver Interface
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import time


class BaseSolver(ABC):
    """Abstract base class for all optimization solvers"""
    
    def __init__(self, name: str):
        self.name = name
        self._available = None
        self._solver_info = None
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if solver is available and ready to use"""
        pass
    
    @abstractmethod
    def get_solver_info(self) -> Dict[str, Any]:
        """Get detailed information about the solver"""
        pass
    
    @abstractmethod
    def solve(self, model_dict: Dict[str, Any], parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], float, Optional[float]]:
        """
        Solve optimization problem
        
        Args:
            model_dict: OMMX model as dictionary (bytes serialized as list)
            parameters: Solver-specific parameters
            
        Returns:
            Tuple of (solution_dict, solve_time, objective_value)
            
        Raises:
            RuntimeError: If solver is not available or solving fails
        """
        pass
    
    def solve_with_timing(self, model_dict: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper that provides consistent response format and error handling
        
        Returns:
            Dictionary with status, objective_value, variables, solve_time, solver, message
        """
        print("DEBUG: solve_with_timing called")
        if not self.is_available():
            return {
                "status": "failed",
                "objective_value": None,
                "variables": {},
                "solve_time": 0.0,
                "solver": self.name,
                "message": f"Solver {self.name} is not available"
            }
        
        try:
            start_time = time.time()
            solution_dict, solve_time, objective_value = self.solve(model_dict, parameters)
            total_time = time.time() - start_time
            print(f"DEBUG: After solve() - solution_dict keys: {list(solution_dict.keys())}")
            
            # Ensure consistent format
            result = {
                "status": solution_dict.get("status", "unknown"),
                "objective_value": objective_value,
                "variables": solution_dict.get("variables", {}),
                "solve_time": solve_time or total_time,
                "solver": self.name,
                "message": solution_dict.get("message", "")
            }
            
            # Preserve OMMX State bytes if present
            print(f"DEBUG: solution_dict keys: {list(solution_dict.keys())}")
            if "ommx_state_bytes" in solution_dict:
                print(f"DEBUG: Found ommx_state_bytes, size: {len(solution_dict['ommx_state_bytes'])}")
                result["ommx_state_bytes"] = solution_dict["ommx_state_bytes"]
            else:
                print("DEBUG: No ommx_state_bytes found in solution_dict")
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            return {
                "status": "failed",
                "objective_value": None,
                "variables": {},
                "solve_time": total_time,
                "solver": self.name,
                "message": f"Solver error: {str(e)}"
            }
    

    
    def __str__(self) -> str:
        return f"{self.name} Solver (available: {self.is_available()})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', available={self.is_available()})>" 