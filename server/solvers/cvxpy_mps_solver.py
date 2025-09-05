"""
CVXPY MPS Solver Implementation

This module provides CVXPY-based MPS solvers using different engines
(CLARABEL, OSQP, CVXOPT, SCS, PROXQP) through CVXPY's solver interface.
Uses comprehensive MPS parser to handle advanced MPS features.
"""

import logging
import time
import os
from typing import Dict, Any, Optional, List
import numpy as np

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None

from .mps_base import BaseMPSSolver
from models import MPSOptimizationResponse, SolverDiagnostics, SolverCapability
from .comprehensive_mps_parser import parse_mps_file
from .mps_converters import to_cvxpy, get_problem_info

logger = logging.getLogger(__name__)


class CVXPYMPSSolver(BaseMPSSolver):
    """Base class for CVXPY MPS solvers with different engines"""
    
    solver_name = None  # To be overridden in subclasses
    
    def __init__(self):
        name = f"CVXPY-{self.solver_name.upper()}"
        super().__init__(name)
        self._available = None
        self._solver_version = "unknown"
        
    def _check_availability(self):
        """Check if CVXPY and the specific solver are available"""
        if not CVXPY_AVAILABLE:
            self._available = False
            logger.warning(f"{self.name} solver not available: cvxpy not installed")
            return
            
        try:
            # Test if the solver is installed by creating a dummy problem
            x = cp.Variable()
            prob = cp.Problem(cp.Minimize(x), [x >= 0])
            
            # Try to solve with this solver
            prob.solve(solver=self.solver_name, verbose=False)
            
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                self._available = True
                # Get CVXPY version
                self._solver_version = f"cvxpy_{cp.__version__}"
                logger.info(f"{self.name} solver available. CVXPY version: {cp.__version__}")
            else:
                self._available = False
                logger.warning(f"{self.name} solver not available: solver test failed with status {prob.status}")
                
        except Exception as e:
            self._available = False
            logger.error(f"Error checking {self.name} availability: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if the solver is available"""
        if self._available is None:
            self._check_availability()
        return self._available or False
    
    def get_capabilities(self) -> List[SolverCapability]:
        """Get solver capabilities - CVXPY solvers support LP and can handle quadratic problems"""
        # All CVXPY solvers support at least LP
        # Many also support quadratic problems, but we'll use LP for compatibility
        return [SolverCapability.LP]
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get CVXPY solver information"""
        info = {
            "name": self.name,
            "available": self.is_available(),
            "capabilities": [cap.value for cap in self.get_capabilities()]
        }
        if self.is_available() and CVXPY_AVAILABLE:
            info["version"] = self._solver_version
            info["cvxpy_version"] = cp.__version__
        else:
            info["version"] = None
            info["cvxpy_version"] = None
        return info
    
    def solve_mps(self, mps_file_path: str, parameters: Dict[str, Any], optimality_tolerance: Optional[float] = None) -> MPSOptimizationResponse:
        """Solve MPS file using CVXPY with the specified solver"""
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
            # Parse MPS file using comprehensive parser and convert to CVXPY
            logger.info(f"Parsing MPS file with comprehensive parser: {mps_file_path}")
            
            mps_problem = parse_mps_file(mps_file_path)
            problem_info = get_problem_info(mps_problem)
            logger.info(f"Comprehensive parser: {problem_info['n_vars']} vars, {problem_info['n_constraints']} constraints, "
                       f"quadratic: {problem_info['has_quadratic']}, integer: {problem_info['has_integer']}")
            
            # Check for unsupported features
            if problem_info['has_integer'] and self.solver_name.lower() not in ['clarabel']:
                logger.warning(f"{self.name}: Integer variables detected but not fully supported - treating as continuous")
            
            cvxpy_problem = to_cvxpy(mps_problem)
            logger.info(f"{self.name}: Using comprehensive parser - full MPS feature support")
            
            # Set up solver options
            solver_options = {}
            
            # Set time limit if provided
            time_limit = parameters.get('max_time')
            if time_limit and time_limit > 0:
                # Different CVXPY solvers have different time limit parameters
                if self.solver_name.lower() == 'clarabel':
                    solver_options['time_limit'] = time_limit
                elif self.solver_name.lower() == 'osqp':
                    solver_options['time_limit'] = time_limit
                elif self.solver_name.lower() == 'cvxopt':
                    solver_options['max_iters'] = min(int(time_limit * 1000), 100000)  # Rough approximation
                elif self.solver_name.lower() == 'scs':
                    solver_options['max_iters'] = min(int(time_limit * 1000), 100000)  # Rough approximation
                elif self.solver_name.lower() == 'proxqp':
                    solver_options['max_iter'] = min(int(time_limit * 1000), 100000)  # Rough approximation
                    
                logger.info(f"{self.name} solver time limit set to {time_limit} seconds.")
            
            # Set optimality tolerance if supported and provided
            if optimality_tolerance is not None:
                if self.solver_name.lower() == 'clarabel':
                    solver_options['tol_feas'] = optimality_tolerance
                    solver_options['tol_gap_abs'] = optimality_tolerance
                elif self.solver_name.lower() == 'osqp':
                    solver_options['eps_abs'] = optimality_tolerance
                    solver_options['eps_rel'] = optimality_tolerance
                elif self.solver_name.lower() == 'cvxopt':
                    solver_options['abstol'] = optimality_tolerance
                    solver_options['reltol'] = optimality_tolerance
                elif self.solver_name.lower() == 'scs':
                    solver_options['eps'] = optimality_tolerance
                elif self.solver_name.lower() == 'proxqp':
                    solver_options['eps_abs'] = optimality_tolerance
                    solver_options['eps_rel'] = optimality_tolerance
                    
                logger.info(f"{self.name} optimality tolerance set to {optimality_tolerance}")
            
            # Set verbose mode
            verbose = parameters.get('verbose', False)
            if not verbose:
                solver_options['verbose'] = False
            
            # Solve
            logger.info(f"{self.name}: Starting optimization")
            start_time = time.time()
            
            try:
                cvxpy_problem.solve(solver=self.solver_name, **solver_options)
                solve_time = time.time() - start_time
                
                # Process results
                if cvxpy_problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    solver_status = "OPTIMAL"
                    objective_value = float(cvxpy_problem.value) if cvxpy_problem.value is not None else 0.0
                    
                    # Extract variable values
                    variables = cvxpy_problem.variables()[0]  # Get the main variable (x)
                    if variables.value is not None:
                        variable_values = {
                            mps_problem.var_names[i]: float(variables.value[i])
                            for i in range(len(mps_problem.var_names))
                        }
                    else:
                        variable_values = {}
                    
                    message = f"{self.name} solved successfully"
                    
                elif cvxpy_problem.status == cp.INFEASIBLE:
                    solver_status = "INFEASIBLE"
                    objective_value = None
                    variable_values = {}
                    message = f"{self.name}: Problem is infeasible"
                    
                elif cvxpy_problem.status == cp.UNBOUNDED:
                    solver_status = "UNBOUNDED"
                    objective_value = None
                    variable_values = {}
                    message = f"{self.name}: Problem is unbounded"
                    
                else:
                    solver_status = "ERROR"
                    objective_value = None
                    variable_values = {}
                    message = f"{self.name}: Solver failed with status {cvxpy_problem.status}"
                
                logger.info(f"{self.name}: Solved in {solve_time:.4f}s with status {solver_status}")
                
                return MPSOptimizationResponse(
                    status=solver_status.lower(),
                    message=message,
                    objective_value=objective_value,
                    variables=variable_values,
                    solve_time=solve_time,
                    solver=self.name,
                    solver_info=SolverDiagnostics(),
                    num_constraints=problem_info['n_constraints'],
                    parameters_used=parameters
                )
                
            except Exception as solve_error:
                solve_time = time.time() - start_time
                logger.error(f"{self.name} solve error: {solve_error}")
                return MPSOptimizationResponse(
                    status="error",
                    message=f"{self.name} solver error: {str(solve_error)}",
                    objective_value=None,
                    variables={},
                    solve_time=solve_time,
                    solver=self.name,
                    solver_info=SolverDiagnostics(),
                    num_constraints=problem_info['n_constraints'],
                    parameters_used=parameters
                )
                
        except Exception as e:
            logger.error(f"{self.name} solver error: {e}")
            return MPSOptimizationResponse(
                status="error",
                message=f"{self.name} solver error: {str(e)}",
                objective_value=None,
                variables={},
                solve_time=0.0,
                solver=self.name,
                solver_info=SolverDiagnostics(),
                num_constraints=0,
                parameters_used=parameters or {}
            )


# Concrete solver implementations
class CVXPYCLARABELSolver(CVXPYMPSSolver):
    """CLARABEL solver via CVXPY - Modern conic optimizer"""
    solver_name = "CLARABEL"


class CVXPYOSQPSolver(CVXPYMPSSolver):
    """OSQP solver via CVXPY - Quadratic programming"""
    solver_name = "OSQP"


class CVXPYCVXOPTSolver(CVXPYMPSSolver):
    """CVXOPT solver via CVXPY - Convex optimization"""
    solver_name = "CVXOPT"


class CVXPYSCSSolver(CVXPYMPSSolver):
    """SCS solver via CVXPY - Splitting conic solver"""
    solver_name = "SCS"


class CVXPYPROXQPSolver(CVXPYMPSSolver):
    """PROXQP solver via CVXPY - Proximal QP solver"""
    solver_name = "PROXQP"


# Additional CVXPY solvers
class CVXPYECOSSolver(CVXPYMPSSolver):
    """ECOS solver via CVXPY - Embedded conic solver"""
    solver_name = "ECOS"


class CVXPYMOSEK(CVXPYMPSSolver):
    """MOSEK solver via CVXPY - Commercial conic solver"""
    solver_name = "MOSEK"


# Factory function for easy solver discovery
def get_available_cvxpy_solvers() -> List[CVXPYMPSSolver]:
    """Get list of all available CVXPY solvers"""
    solver_classes = [
        CVXPYCLARABELSolver,
        CVXPYOSQPSolver,
        CVXPYCVXOPTSolver,
        CVXPYSCSSolver,
        CVXPYPROXQPSolver,
        CVXPYECOSSolver,
        CVXPYMOSEK,
    ]
    
    available_solvers = []
    for solver_class in solver_classes:
        solver = solver_class()
        if solver.is_available():
            available_solvers.append(solver)
    
    return available_solvers