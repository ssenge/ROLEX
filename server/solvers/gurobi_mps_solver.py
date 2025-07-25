"""
ROLEX Server - Gurobi MPS Solver
Native MPS file solving with Gurobi
"""
import time
import logging
from typing import Dict, Any

from .mps_base import BaseMPSSolver
from models import MPSOptimizationResponse, SolverDiagnostics

# Import dependencies with proper error handling
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

logger = logging.getLogger(__name__)


class GurobiMPSSolver(BaseMPSSolver):
    """Gurobi solver with native MPS file support"""
    
    def __init__(self):
        super().__init__("Gurobi-MPS")
        
    def is_available(self) -> bool:
        """Check if Gurobi is available"""
        return GUROBI_AVAILABLE
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get Gurobi solver information"""
        info = {
            "name": "Gurobi with Native MPS Support",
            "available": self.is_available(),
            "capabilities": ["linear", "quadratic", "mixed-integer", "mps"]
        }
        
        if not GUROBI_AVAILABLE:
            info["status"] = "Gurobi not available"
        else:
            try:
                # Get Gurobi version
                info["version"] = f"{gp.gurobi.version()[0]}.{gp.gurobi.version()[1]}"
                info["status"] = "available"
            except Exception as e:
                info["status"] = f"error: {str(e)}"
        
        return info
    
    def solve_mps(self, mps_file_path: str, parameters: Dict[str, Any]) -> MPSOptimizationResponse:
        """
        Solve MPS file using Gurobi
        
        Args:
            mps_file_path: Path to MPS file
            parameters: Solver parameters
            
        Returns:
            MPSOptimizationResponse with solution
        """
        if not self.is_available():
            raise RuntimeError("Gurobi solver is not available")
        
        # Validate parameters
        validated_params = self._validate_parameters(parameters)
        
        try:
            # Read MPS file
            model = gp.read(mps_file_path)
            
            # Set parameters
            self._set_gurobi_parameters(model, validated_params)
            
            # Solve
            start_time = time.time()
            model.optimize()
            solve_time = time.time() - start_time
            
            # Process results
            return self._process_gurobi_results(model, solve_time, validated_params)
            
        except Exception as e:
            logger.error(f"Gurobi MPS solve failed: {str(e)}")
            raise RuntimeError(f"Gurobi solve failed: {str(e)}")
    
    def _set_gurobi_parameters(self, model: gp.Model, parameters: Dict[str, Any]):
        """Set Gurobi parameters from validated parameters"""
        
        if 'max_time' in parameters:
            model.setParam('TimeLimit', parameters['max_time'])
            logger.info(f"Set TimeLimit to {parameters['max_time']} seconds")
        
        if 'threads' in parameters:
            model.setParam('Threads', parameters['threads'])
            logger.info(f"Set Threads to {parameters['threads']}")
        
        if 'gap_tolerance' in parameters:
            model.setParam('MIPGap', parameters['gap_tolerance'])
            logger.info(f"Set MIPGap to {parameters['gap_tolerance']}")
        
        if 'verbose' in parameters:
            if parameters['verbose']:
                model.setParam('OutputFlag', 1)
            else:
                model.setParam('OutputFlag', 0)
    
    def _process_gurobi_results(self, model: gp.Model, solve_time: float, parameters: Dict[str, Any]) -> MPSOptimizationResponse:
        """Process Gurobi optimization results"""
        
        # Map Gurobi status to our status
        status_map = {
            GRB.OPTIMAL: "optimal",
            GRB.INFEASIBLE: "infeasible",
            GRB.UNBOUNDED: "unbounded",
            GRB.INF_OR_UNBD: "infeasible_or_unbounded",
            GRB.TIME_LIMIT: "time_limit",
            GRB.NODE_LIMIT: "node_limit",
            GRB.SOLUTION_LIMIT: "solution_limit",
            GRB.INTERRUPTED: "interrupted",
            GRB.NUMERIC: "numeric_error",
            GRB.SUBOPTIMAL: "suboptimal"
        }
        
        status = status_map.get(model.Status, "unknown")
        
        # Get objective value
        objective_value = None
        if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
            try:
                objective_value = model.objVal
            except:
                objective_value = None
        
        # Get variables
        variables = {}
        if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
            try:
                for var in model.getVars():
                    variables[var.varName] = var.x
            except Exception as e:
                logger.warning(f"Could not extract variables: {str(e)}")
        
        # Get solver diagnostics
        solver_info = SolverDiagnostics()
        try:
            solver_info.iterations = int(model.getAttr('IterCount'))
        except:
            pass
        
        try:
            solver_info.nodes = int(model.getAttr('NodeCount'))
        except:
            pass
        
        try:
            solver_info.gap = float(model.getAttr('MIPGap'))
        except:
            pass
        
        try:
            solver_info.bound = float(model.getAttr('ObjBound'))
        except:
            pass
        
        solver_info.status_code = model.Status
        
        # Generate message
        message = f"Gurobi status: {status}"
        if objective_value is not None:
            message += f", objective: {objective_value}"
        
        return MPSOptimizationResponse(
            status=status,
            objective_value=objective_value,
            variables=variables,
            solve_time=solve_time,
            solver=self.name,
            message=message,
            solver_info=solver_info
        )
    
    def _validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Gurobi-specific parameters"""
        validated = super()._validate_parameters(parameters)
        
        # Add Gurobi-specific parameter validation
        if 'feasibility_tolerance' in parameters:
            tol = parameters['feasibility_tolerance']
            if isinstance(tol, (int, float)) and tol > 0:
                validated['feasibility_tolerance'] = float(tol)
        
        if 'optimality_tolerance' in parameters:
            tol = parameters['optimality_tolerance']
            if isinstance(tol, (int, float)) and tol > 0:
                validated['optimality_tolerance'] = float(tol)
        
        return validated 