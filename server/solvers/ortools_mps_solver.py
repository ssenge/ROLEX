"""
OR-Tools MPS Solver Implementation

This module provides OR-Tools-based MPS solvers using different engines
(GLOP, CBC, CLP, SCIP) that are bundled with OR-Tools.
Uses model_builder.ModelSolver(engine_name) as agreed.
"""

import logging
import time
from typing import Dict, Any, Optional, List

try:
    from ortools.linear_solver.python import model_builder
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    model_builder = None

from .mps_base import BaseMPSSolver
from models import MPSOptimizationResponse, SolverDiagnostics, SolverCapability

logger = logging.getLogger(__name__)


class ORToolsMPSSolver(BaseMPSSolver):
    """Base class for OR-Tools MPS solvers with different engines"""
    
    engine_name = None  # To be overridden in subclasses
    
    def __init__(self):
        self.name = f"OR-Tools-{self.engine_name}"
        self._available = None
        self._solver_version = "unknown"
        
    def _check_availability(self):
        """Check if OR-Tools and the specific engine are available"""
        if not ORTOOLS_AVAILABLE:
            self._available = False
            logger.warning(f"{self.name} solver not available: ortools not installed")
            return
            
        try:
            # Test if we can create a solver with this engine using model_builder
            solver = model_builder.ModelSolver(self.engine_name)
            self._available = True
            
            # Get OR-Tools version
            try:
                import ortools
                self._solver_version = ortools.__version__
            except ImportError:
                self._solver_version = "ortools_version_unknown"
                
            logger.info(f"{self.name} solver available. OR-Tools version: {self._solver_version}")
        except Exception as e:
            self._available = False
            logger.error(f"Error checking {self.name} availability: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if the solver is available"""
        if self._available is None:
            self._check_availability()
        return self._available or False
    
    def solve_mps(self, mps_file_path: str, parameters: Dict[str, Any], optimality_tolerance: Optional[float] = None) -> MPSOptimizationResponse:
        """Solve MPS file using OR-Tools model_builder with the specified engine"""
        if not self.is_available():
            return MPSOptimizationResponse(
                status="failed",
                message=f"{self.name} solver not available",
                solve_time=0.0,
                solver=self.name,
                solver_info=SolverDiagnostics(),
                parameters_used=parameters
            )
        
        try:
            # Create model and import MPS file using model_builder (as we agreed!)
            model = model_builder.ModelBuilder()
            logger.info(f"Loading MPS file: {mps_file_path}")
            model.import_from_mps_file(mps_file_path)
            
            # Log model info for debugging
            logger.info(f"{self.engine_name}: Model loaded - {model.num_variables} variables, {model.num_constraints} constraints")
            
            # Create solver with specific engine using ModelSolver(engine_name)
            solver = model_builder.ModelSolver(self.engine_name)
            logger.info(f"{self.engine_name}: Created solver instance")
            
            # Set time limit if provided
            time_limit = parameters.get('max_time')
            if time_limit and time_limit > 0:
                solver.set_time_limit_in_seconds(time_limit)
                logger.info(f"{self.engine_name} solver time limit set to {time_limit} seconds.")
            
            # Set optimality tolerance if supported and provided
            if optimality_tolerance is not None:
                # Note: Not all OR-Tools engines support this, but we try
                try:
                    if hasattr(solver, 'set_primal_tolerance'):
                        solver.set_primal_tolerance(optimality_tolerance)
                    elif hasattr(solver, 'set_dual_tolerance'):
                        solver.set_dual_tolerance(optimality_tolerance)
                except Exception as e:
                    logger.warning(f"Could not set optimality tolerance for {self.engine_name}: {e}")
            
            # Solve
            logger.info(f"{self.engine_name}: Starting solve...")
            start_time = time.time()
            status = solver.solve(model)
            solve_time = time.time() - start_time
            logger.info(f"{self.engine_name}: Solve completed with status: {status} in {solve_time:.4f}s")
            
            # Extract results
            objective_value = None
            variable_values = {}
            solver_status = "unknown"
            
            if status == model_builder.SolveStatus.OPTIMAL:
                solver_status = "OPTIMAL"
                objective_value = solver.objective_value
                for i in range(model.num_variables):
                    var = model.var_from_index(i)
                    variable_values[var.name] = solver.value(var)
            elif status == model_builder.SolveStatus.FEASIBLE:
                solver_status = "FEASIBLE"
                objective_value = solver.objective_value
                for i in range(model.num_variables):
                    var = model.var_from_index(i)
                    variable_values[var.name] = solver.value(var)
            elif status == model_builder.SolveStatus.INFEASIBLE:
                solver_status = "INFEASIBLE"
            elif status == model_builder.SolveStatus.UNBOUNDED:
                solver_status = "UNBOUNDED"
            elif status == model_builder.SolveStatus.MODEL_INVALID:
                solver_status = "ERROR"
                logger.error(f"{self.engine_name}: Model is invalid.")
            elif status == model_builder.SolveStatus.NOT_SOLVED:
                # Check if time limit was reached
                if time_limit and solve_time >= time_limit * 0.99:
                    solver_status = "TIMELIMIT_REACHED"
                    logger.warning(f"{self.engine_name}: Time limit reached, no solution found.")
                else:
                    solver_status = "ERROR"
                    logger.warning(f"{self.engine_name}: Problem not solved.")
            elif status == model_builder.SolveStatus.ABNORMAL:
                solver_status = "ERROR"
                logger.error(f"{self.engine_name}: Abnormal termination.")
            elif status == model_builder.SolveStatus.UNKNOWN_STATUS:
                solver_status = "ERROR"
                logger.error(f"{self.engine_name}: Unknown status - this may indicate: 1) solver doesn't support this problem type, 2) solver configuration issue, or 3) internal solver error. Try a different solver like GLOP for LP problems.")
            else:
                solver_status = "ERROR"
                logger.error(f"{self.engine_name}: Unexpected solve status: {status}")
            
            # Check if time limit was reached even if a feasible/optimal solution was found
            if time_limit and solve_time >= time_limit * 0.99 and solver_status in ['OPTIMAL', 'FEASIBLE']:
                solver_status = "TIMELIMIT_REACHED"
                logger.warning(f"{self.engine_name}: Time limit reached, returning best found solution.")
            
            success = solver_status in ['OPTIMAL', 'FEASIBLE']
            message = f"Solved with {self.engine_name}"
            if not success:
                message = f"{self.engine_name} solver finished with status: {solver_status}"
            
            return MPSOptimizationResponse(
                status=solver_status.lower(),  # Convert to lowercase to match expected format
                message=message,
                objective_value=objective_value,
                variables=variable_values,
                solve_time=solve_time,
                solver=self.name,
                solver_info=SolverDiagnostics(),  # OR-Tools model_builder doesn't expose detailed diagnostics
                num_constraints=model.num_constraints,
                parameters_used=parameters
            )
            
        except Exception as e:
            logger.error(f"Error solving with {self.name}: {str(e)}")
            return MPSOptimizationResponse(
                status="failed",
                message=f"Error solving with {self.name}: {str(e)}",
                solve_time=0.0,
                solver=self.name,
                solver_info=SolverDiagnostics(),
                parameters_used=parameters
            )


class ORToolsGLOPSolver(ORToolsMPSSolver):
    """OR-Tools GLOP (Linear Programming) solver"""
    engine_name = "GLOP"
    
    def get_capabilities(self) -> List[SolverCapability]:
        """GLOP is a pure LP solver"""
        return [SolverCapability.LP]
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get GLOP solver information"""
        info = {
            "name": "OR-Tools GLOP",
            "available": self.is_available(),
            "capabilities": [cap.value for cap in self.get_capabilities()]
        }
        
        # Add OR-Tools version if available
        if self.is_available():
            info["version"] = self._solver_version
        else:
            info["version"] = None
            
        return info


class ORToolsCBCSolver(ORToolsMPSSolver):
    """OR-Tools CBC (Mixed-Integer Programming) solver"""
    engine_name = "CBC"
    
    def get_capabilities(self) -> List[SolverCapability]:
        """CBC can solve LP and MIP problems"""
        return [SolverCapability.LP, SolverCapability.MIP]
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get CBC solver information"""
        info = {
            "name": "OR-Tools CBC",
            "available": self.is_available(),
            "capabilities": [cap.value for cap in self.get_capabilities()]
        }
        
        # Add OR-Tools version if available
        if self.is_available():
            info["version"] = self._solver_version
        else:
            info["version"] = None
            
        return info


class ORToolsCLPSolver(ORToolsMPSSolver):
    """OR-Tools CLP (Linear Programming) solver"""
    engine_name = "CLP"
    
    def get_capabilities(self) -> List[SolverCapability]:
        """CLP is a pure LP solver"""
        return [SolverCapability.LP]
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get CLP solver information"""
        info = {
            "name": "OR-Tools CLP",
            "available": self.is_available(),
            "capabilities": [cap.value for cap in self.get_capabilities()]
        }
        
        # Add OR-Tools version if available
        if self.is_available():
            info["version"] = self._solver_version
        else:
            info["version"] = None
            
        return info


class ORToolsSCIPSolver(ORToolsMPSSolver):
    """OR-Tools SCIP (Mixed-Integer Programming) solver"""
    engine_name = "SCIP"
    
    def get_capabilities(self) -> List[SolverCapability]:
        """SCIP can solve LP and MIP problems"""
        return [SolverCapability.LP, SolverCapability.MIP]
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get SCIP solver information"""
        info = {
            "name": "OR-Tools SCIP",
            "available": self.is_available(),
            "capabilities": [cap.value for cap in self.get_capabilities()]
        }
        
        # Add OR-Tools version if available
        if self.is_available():
            info["version"] = self._solver_version
        else:
            info["version"] = None
            
        return info


class ORToolsPDLPSolver(ORToolsMPSSolver):
    """OR-Tools PDLP (Primal-Dual Linear Programming) solver"""
    engine_name = "PDLP"
    
    def get_capabilities(self) -> List[SolverCapability]:
        """PDLP can solve LP problems"""
        return [SolverCapability.LP]
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get PDLP solver information"""
        info = {
            "name": "OR-Tools PDLP",
            "available": self.is_available(),
            "capabilities": [cap.value for cap in self.get_capabilities()]
        }
        
        # Add OR-Tools version if available
        if self.is_available():
            info["version"] = self._solver_version
        else:
            info["version"] = None
            
        return info 