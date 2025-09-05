"""
Pyomo MPS Solver Implementation

This module provides Pyomo-based MPS solvers using different engines
(CPLEX, Gurobi, GLPK, CBC, IPOPT, SCIP, HiGHS) through Pyomo's SolverFactory.
Uses MPS-to-Pyomo converter to handle MPS files.
"""

import logging
import time
import os
from typing import Dict, Any, Optional, List

try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory
    from pyomo.core.base.PyomoModel import ConcreteModel
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    pyo = None
    SolverFactory = None

from .mps_base import BaseMPSSolver
from models import MPSOptimizationResponse, SolverDiagnostics, SolverCapability
from .comprehensive_mps_parser import parse_mps_file
from .mps_converters import to_pyomo, get_problem_info

logger = logging.getLogger(__name__)


class PyomoMPSSolver(BaseMPSSolver):
    """Base class for Pyomo MPS solvers with different engines"""
    
    solver_name = None  # To be overridden in subclasses
    
    def __init__(self):
        name = f"Pyomo-{self.solver_name.upper()}"
        super().__init__(name)
        self._available = None
        self._solver_version = "unknown"
        self._solver_factory = None
        
    def _check_availability(self):
        """Check if Pyomo and the specific solver are available"""
        if not PYOMO_AVAILABLE:
            self._available = False
            logger.warning(f"{self.name} solver not available: pyomo not installed")
            return
            
        try:
            # Test if we can create a solver with this engine using SolverFactory
            self._solver_factory = SolverFactory(self.solver_name)
            self._available = self._solver_factory.available()
            
            if self._available:
                # Get solver version if possible
                try:
                    if hasattr(self._solver_factory, 'version'):
                        version_info = self._solver_factory.version()
                        if version_info:
                            self._solver_version = str(version_info)
                except Exception:
                    self._solver_version = f"{self.solver_name}_version_unknown"
                    
                logger.info(f"{self.name} solver available. Version: {self._solver_version}")
            else:
                logger.warning(f"{self.name} solver not available: {self.solver_name} not found or not properly configured")
                
        except Exception as e:
            self._available = False
            logger.error(f"Error checking {self.name} availability: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if the solver is available"""
        if self._available is None:
            self._check_availability()
        return self._available or False
    
    def solve_mps(self, mps_file_path: str, parameters: Dict[str, Any], optimality_tolerance: Optional[float] = None) -> MPSOptimizationResponse:
        """Solve MPS file using Pyomo with the specified solver"""
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
            # Parse MPS file using comprehensive parser and convert to Pyomo
            logger.info(f"Parsing MPS file with comprehensive parser: {mps_file_path}")
            
            mps_problem = parse_mps_file(mps_file_path)
            problem_info = get_problem_info(mps_problem)
            logger.info(f"Comprehensive parser: {problem_info['n_vars']} vars, {problem_info['n_constraints']} constraints, "
                       f"quadratic: {problem_info['has_quadratic']}, integer: {problem_info['has_integer']}")
            
            model = to_pyomo(mps_problem)
            logger.info(f"{self.solver_name}: Using comprehensive parser - full MPS feature support")
            
            # Create solver instance
            solver = SolverFactory(self.solver_name)
            logger.info(f"{self.solver_name}: Created solver instance")
            
            # Set solver options from parameters
            solver_options = {}
            
            # Set time limit if provided
            time_limit = parameters.get('max_time')
            if time_limit and time_limit > 0:
                # Different solvers have different time limit parameter names
                if self.solver_name.lower() in ['gurobi']:
                    solver_options['TimeLimit'] = time_limit
                elif self.solver_name.lower() in ['cplex']:
                    solver_options['timelimit'] = time_limit
                elif self.solver_name.lower() in ['glpk']:
                    solver_options['tmlim'] = time_limit
                elif self.solver_name.lower() in ['cbc']:
                    solver_options['seconds'] = time_limit
                elif self.solver_name.lower() in ['ipopt']:
                    solver_options['max_cpu_time'] = time_limit
                elif self.solver_name.lower() in ['scip']:
                    solver_options['limits/time'] = time_limit
                elif self.solver_name.lower() in ['highs']:
                    solver_options['time_limit'] = time_limit
                    
                logger.info(f"{self.solver_name} solver time limit set to {time_limit} seconds.")
            
            # Set optimality tolerance if supported and provided
            if optimality_tolerance is not None:
                if self.solver_name.lower() in ['gurobi']:
                    solver_options['MIPGap'] = optimality_tolerance
                    solver_options['OptimalityTol'] = optimality_tolerance
                elif self.solver_name.lower() in ['cplex']:
                    solver_options['mip/tolerances/mipgap'] = optimality_tolerance
                    solver_options['simplex/tolerances/optimality'] = optimality_tolerance
                elif self.solver_name.lower() in ['cbc']:
                    solver_options['ratio'] = optimality_tolerance
                elif self.solver_name.lower() in ['ipopt']:
                    solver_options['tol'] = optimality_tolerance
                    
                logger.info(f"Set optimality tolerance: {optimality_tolerance}")
            
            # Apply solver options
            for option, value in solver_options.items():
                solver.options[option] = value
            
            # Set additional parameters from parameters dict
            # Skip parameters that are handled elsewhere or aren't solver options
            skip_params = ['max_time', 'verbose']
            for key, value in parameters.items():
                if key not in skip_params:
                    solver.options[key] = value
            
            # Solve
            logger.info(f"{self.solver_name}: Starting solve...")
            start_time = time.time()
            result = solver.solve(model, tee=parameters.get('verbose', False))
            solve_time = time.time() - start_time
            logger.info(f"{self.solver_name}: Solve completed in {solve_time:.4f}s")
            
            # Extract results
            objective_value = None
            variable_values = {}
            solver_status = "unknown"
            
            # Check termination condition
            termination_condition = result.solver.termination_condition
            
            if termination_condition == pyo.TerminationCondition.optimal:
                solver_status = "optimal"
                try:
                    objective_value = pyo.value(model.objective)
                    # Extract variable values using original variable names
                    for i, var_name in enumerate(model.var_names):
                        variable_values[var_name] = pyo.value(model.x[i])
                except Exception as e:
                    logger.warning(f"Could not extract objective/variables from optimal solution: {e}")
                        
            elif termination_condition == pyo.TerminationCondition.feasible:
                solver_status = "feasible"
                try:
                    objective_value = pyo.value(model.objective)
                    # Extract variable values
                    for i, var_name in enumerate(model.var_names):
                        variable_values[var_name] = pyo.value(model.x[i])
                except Exception as e:
                    logger.warning(f"Could not extract objective/variables from feasible solution: {e}")
                    
            elif termination_condition == pyo.TerminationCondition.infeasible:
                solver_status = "infeasible"
            elif termination_condition == pyo.TerminationCondition.unbounded:
                solver_status = "unbounded"
            elif termination_condition == pyo.TerminationCondition.maxTimeLimit:
                solver_status = "timelimit_reached"
                # Try to get best solution if available
                try:
                    objective_value = pyo.value(model.objective)
                    for i, var_name in enumerate(model.var_names):
                        variable_values[var_name] = pyo.value(model.x[i])
                except Exception as e:
                    logger.warning(f"Time limit reached, no solution available: {e}")
            elif termination_condition == pyo.TerminationCondition.maxIterations:
                solver_status = "timelimit_reached"
            else:
                solver_status = "error"
                logger.error(f"{self.solver_name}: Termination condition: {termination_condition}")
            
            success = solver_status in ['optimal', 'feasible']
            message = f"Solved with {self.solver_name}"
            if not success:
                message = f"{self.solver_name} solver finished with status: {solver_status}"
            
            return MPSOptimizationResponse(
                status=solver_status.lower(),
                message=message,
                objective_value=objective_value,
                variables=variable_values,
                solve_time=solve_time,
                solver=self.name,
                solver_info=SolverDiagnostics(),
                num_constraints=model_info['num_constraints'],
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
            
        


class PyomoCPLEXSolver(PyomoMPSSolver):
    """Pyomo CPLEX solver"""
    solver_name = "cplex"
    
    def get_capabilities(self) -> List[SolverCapability]:
        """CPLEX can solve LP and MIP problems"""
        return [SolverCapability.LP, SolverCapability.MIP]
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get CPLEX solver information"""
        return {
            "name": "Pyomo CPLEX",
            "available": self.is_available(),
            "capabilities": [cap.value for cap in self.get_capabilities()],
            "version": self._solver_version if self.is_available() else None
        }


class PyomoGurobiSolver(PyomoMPSSolver):
    """Pyomo Gurobi solver"""
    solver_name = "gurobi"
    
    def get_capabilities(self) -> List[SolverCapability]:
        """Gurobi can solve LP and MIP problems"""
        return [SolverCapability.LP, SolverCapability.MIP]
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get Gurobi solver information"""
        return {
            "name": "Pyomo Gurobi",
            "available": self.is_available(),
            "capabilities": [cap.value for cap in self.get_capabilities()],
            "version": self._solver_version if self.is_available() else None
        }


class PyomoGLPKSolver(PyomoMPSSolver):
    """Pyomo GLPK solver"""
    solver_name = "glpk"
    
    def get_capabilities(self) -> List[SolverCapability]:
        """GLPK can solve LP and MIP problems"""
        return [SolverCapability.LP, SolverCapability.MIP]
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get GLPK solver information"""
        return {
            "name": "Pyomo GLPK",
            "available": self.is_available(),
            "capabilities": [cap.value for cap in self.get_capabilities()],
            "version": self._solver_version if self.is_available() else None
        }


class PyomoCBCSolver(PyomoMPSSolver):
    """Pyomo CBC solver"""
    solver_name = "cbc"
    
    def get_capabilities(self) -> List[SolverCapability]:
        """CBC can solve LP and MIP problems"""
        return [SolverCapability.LP, SolverCapability.MIP]
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get CBC solver information"""
        return {
            "name": "Pyomo CBC",
            "available": self.is_available(),
            "capabilities": [cap.value for cap in self.get_capabilities()],
            "version": self._solver_version if self.is_available() else None
        }


class PyomoIPOPTSolver(PyomoMPSSolver):
    """Pyomo IPOPT solver"""
    solver_name = "ipopt"
    
    def get_capabilities(self) -> List[SolverCapability]:
        """IPOPT is for nonlinear programming, but can handle LP problems from MPS files"""
        return [SolverCapability.LP]
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get IPOPT solver information"""
        return {
            "name": "Pyomo IPOPT",
            "available": self.is_available(),
            "capabilities": [cap.value for cap in self.get_capabilities()],
            "version": self._solver_version if self.is_available() else None
        }


class PyomoSCIPSolver(PyomoMPSSolver):
    """Pyomo SCIP solver"""
    solver_name = "scip"
    
    def get_capabilities(self) -> List[SolverCapability]:
        """SCIP can solve LP and MIP problems"""
        return [SolverCapability.LP, SolverCapability.MIP]
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get SCIP solver information"""
        return {
            "name": "Pyomo SCIP",
            "available": self.is_available(),
            "capabilities": [cap.value for cap in self.get_capabilities()],
            "version": self._solver_version if self.is_available() else None
        }


class PyomoHiGHSSolver(PyomoMPSSolver):
    """Pyomo HiGHS solver"""
    solver_name = "highs"
    
    def get_capabilities(self) -> List[SolverCapability]:
        """HiGHS can solve LP and MIP problems"""
        return [SolverCapability.LP, SolverCapability.MIP]
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get HiGHS solver information"""
        return {
            "name": "Pyomo HiGHS",
            "available": self.is_available(),
            "capabilities": [cap.value for cap in self.get_capabilities()],
            "version": self._solver_version if self.is_available() else None
        }