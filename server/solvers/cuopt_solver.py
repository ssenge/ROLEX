"""
ROLEX Server - cuOpt Solver
GPU-accelerated optimization solver with OMMX integration
"""
import time
import logging
import json
from typing import Dict, Any, Tuple, Optional, List, TYPE_CHECKING
import numpy as np

from .base import BaseSolver

# Import dependencies with proper error handling
try:
    import ommx.v1 as ommx
    OMMX_AVAILABLE = True
except ImportError:
    OMMX_AVAILABLE = False
    ommx = None

# Type checking imports
if TYPE_CHECKING:
    from ommx.v1 import Instance

# Don't import cuopt at module level - do lazy import
# This avoids library path issues during server startup
CUOPT_AVAILABLE = None  # Will be determined lazily

# Check GPU availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class CuOptSolver(BaseSolver):
    """cuOpt solver with OMMX integration - GPU-only"""
    
    def __init__(self):
        super().__init__("cuOpt")
        self.version = "25.5.1"
        
    def is_available(self) -> bool:
        """Check if cuOpt is available with GPU support"""
        # Lazy import check
        cuopt_available = self._check_cuopt_import()
        logger.info(f"cuOpt import check: {cuopt_available}")
        if not cuopt_available:
            logger.info("cuOpt library not available")
            return False
            
        logger.info(f"OMMX_AVAILABLE: {OMMX_AVAILABLE}")
        if not OMMX_AVAILABLE:
            logger.info("OMMX library not available")
            return False
            
        # Check GPU availability
        gpu_available = self._check_gpu_availability()
        logger.info(f"GPU availability check: {gpu_available}")
        if not gpu_available:
            logger.info("No compatible GPU found for cuOpt")
            return False
            
        logger.info("cuOpt solver is available!")
        return True
        
    def _check_cuopt_import(self) -> bool:
        """Lazy import check for cuOpt"""
        global CUOPT_AVAILABLE
        if CUOPT_AVAILABLE is None:
            try:
                import cuopt
                CUOPT_AVAILABLE = True
                logger.info("cuOpt lazy import successful")
            except ImportError as e:
                CUOPT_AVAILABLE = False
                logger.info(f"cuOpt lazy import failed: {e}")
        else:
            logger.info(f"cuOpt import already checked: {CUOPT_AVAILABLE}")
        return CUOPT_AVAILABLE
        
    def _check_gpu_availability(self) -> bool:
        """Check if compatible GPU is available"""
        if not TORCH_AVAILABLE:
            return False
            
        try:
            # Check if CUDA is available
            if not torch.cuda.is_available():
                return False
                
            # Check GPU compute capability (need >= 7.0 for cuOpt)
            device_count = torch.cuda.device_count()
            if device_count == 0:
                return False
                
            for i in range(device_count):
                major, minor = torch.cuda.get_device_capability(i)
                compute_capability = float(f"{major}.{minor}")
                if compute_capability >= 7.0:
                    return True
                    
            return False
            
        except Exception as e:
            logger.debug(f"GPU check failed: {e}")
            return False
            
    def get_solver_info(self) -> Dict[str, Any]:
        """Get detailed solver information"""
        info = {
            "name": self.name,
            "version": self.version,
            "available": self.is_available(),
            "gpu_required": True,
            "supported_problem_types": ["LP", "MILP", "MIQP"],
            "features": [
                "GPU-accelerated solving",
                "Large-scale optimization",
                "Mixed-integer programming", 
                "Quadratic programming",
                "Parallel processing"
            ]
        }
        
        # Add GPU information if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(i)
                gpu_info.append({
                    "device": i,
                    "name": torch.cuda.get_device_name(i),
                    "compute_capability": f"{major}.{minor}",
                    "compatible": float(f"{major}.{minor}") >= 7.0
                })
            info["gpu_devices"] = gpu_info
            
        return info
        
    def solve(self, model_dict: Dict[str, Any], parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], float, Optional[float]]:
        """Solve optimization problem using cuOpt"""
        if not self.is_available():
            gpu_info = self.get_solver_info()
            raise RuntimeError(f"cuOpt solver not available - requires GPU with compute capability >= 7.0. GPU info: {gpu_info}")
            
        try:
            start_time = time.time()
            
            # Convert OMMX bytes to OMMX instance using public API
            ommx_bytes = bytes(model_dict)
            ommx_instance = ommx.Instance.from_bytes(ommx_bytes)
            
            # Convert OMMX instance to cuOpt model
            cuopt_model = self._convert_ommx_to_cuopt(ommx_instance)
            
            # Apply solver parameters
            self._apply_parameters(cuopt_model, parameters)
            
            # Solve with cuOpt
            solution = self._solve_with_cuopt(cuopt_model)
            
            # Convert solution back to OMMX format
            result = self._convert_solution_to_ommx(solution, ommx_instance)
            
            solve_time = time.time() - start_time
            objective_value = result.get("objective_value")
            
            return result, solve_time, objective_value
            
        except Exception as e:
            logger.error(f"cuOpt solve failed: {e}")
            raise RuntimeError(f"cuOpt solve failed: {str(e)}")
            
    def _apply_parameters(self, model, parameters: Dict[str, Any]):
        """Apply solver parameters to cuOpt model"""
        try:
            # Default parameters
            model.set_parameter('device', 'gpu')
            model.set_parameter('presolve', True)
            model.set_parameter('time_limit', 300)  # 5 minutes default
            
            # Apply user-provided parameters
            if 'time_limit' in parameters:
                model.set_parameter('time_limit', parameters['time_limit'])
                
            if 'verbose' in parameters:
                model.set_parameter('verbose', parameters['verbose'])
                
            if 'threads' in parameters:
                model.set_parameter('threads', parameters['threads'])
                
            if 'gap_tolerance' in parameters:
                model.set_parameter('gap_tolerance', parameters['gap_tolerance'])
                
            if 'feasibility_tolerance' in parameters:
                model.set_parameter('feasibility_tolerance', parameters['feasibility_tolerance'])
                
            logger.info(f"Applied cuOpt parameters: {parameters}")
            
        except Exception as e:
            logger.warning(f"Failed to apply some parameters: {e}")
            
    def _convert_ommx_to_cuopt(self, ommx_instance: "Instance") -> Any:
        """Convert OMMX instance to cuOpt model"""
        try:
            # Import cuOpt at runtime
            import cuopt
            
            # Create cuOpt model
            model = cuopt.Model()
            
            # Add decision variables
            cuopt_vars = {}
            for var in ommx_instance.decision_variables:
                if var.kind == ommx.DecisionVariable.Kind.CONTINUOUS:
                    cuopt_var = model.add_variable(
                        name=var.name,
                        lower_bound=var.lower_bound if hasattr(var, 'lower_bound') else None,
                        upper_bound=var.upper_bound if hasattr(var, 'upper_bound') else None,
                        var_type='continuous'
                    )
                elif var.kind == ommx.DecisionVariable.Kind.INTEGER:
                    cuopt_var = model.add_variable(
                        name=var.name,
                        lower_bound=var.lower_bound if hasattr(var, 'lower_bound') else None,
                        upper_bound=var.upper_bound if hasattr(var, 'upper_bound') else None,
                        var_type='integer'
                    )
                elif var.kind == ommx.DecisionVariable.Kind.BINARY:
                    cuopt_var = model.add_variable(
                        name=var.name,
                        lower_bound=0,
                        upper_bound=1,
                        var_type='binary'
                    )
                else:
                    raise ValueError(f"Unsupported variable type: {var.kind}")
                    
                cuopt_vars[var.id] = cuopt_var
                
            # Add constraints
            for constraint in ommx_instance.constraints:
                # Convert OMMX constraint to cuOpt constraint
                expr = self._convert_linear_expression(constraint.function, cuopt_vars)
                
                if constraint.equality == ommx.Constraint.LESS_THAN_OR_EQUAL_TO_ZERO:
                    model.add_constraint(expr <= 0, name=constraint.name)
                elif constraint.equality == ommx.Constraint.EQUAL_TO_ZERO:
                    model.add_constraint(expr == 0, name=constraint.name)
                elif constraint.equality == ommx.Constraint.GREATER_THAN_OR_EQUAL_TO_ZERO:
                    model.add_constraint(expr >= 0, name=constraint.name)
                else:
                    raise ValueError(f"Unsupported constraint type: {constraint.equality}")
                    
            # Set objective
            obj_expr = self._convert_linear_expression(ommx_instance.objective.function, cuopt_vars)
            
            if ommx_instance.sense == ommx.Instance.MINIMIZE:
                model.set_objective(obj_expr, sense='minimize')
            else:
                model.set_objective(obj_expr, sense='maximize')
                
            return model
            
        except Exception as e:
            logger.error(f"OMMX to cuOpt conversion failed: {e}")
            raise
            
    def _convert_linear_expression(self, expr, cuopt_vars):
        """Convert OMMX linear expression to cuOpt expression"""
        try:
            # Handle different expression types
            if hasattr(expr, 'linear'):
                # Linear expression
                cuopt_expr = 0
                for term in expr.linear.terms:
                    var_id = term.variable_id
                    coeff = term.coefficient
                    if var_id in cuopt_vars:
                        cuopt_expr += coeff * cuopt_vars[var_id]
                
                # Add constant term
                if hasattr(expr.linear, 'constant'):
                    cuopt_expr += expr.linear.constant
                    
                return cuopt_expr
                
            elif hasattr(expr, 'quadratic'):
                # Quadratic expression (if supported)
                cuopt_expr = 0
                
                # Linear terms
                for term in expr.quadratic.linear_terms:
                    var_id = term.variable_id
                    coeff = term.coefficient
                    if var_id in cuopt_vars:
                        cuopt_expr += coeff * cuopt_vars[var_id]
                
                # Quadratic terms
                for term in expr.quadratic.quadratic_terms:
                    var_id1 = term.variable_id_1
                    var_id2 = term.variable_id_2
                    coeff = term.coefficient
                    if var_id1 in cuopt_vars and var_id2 in cuopt_vars:
                        cuopt_expr += coeff * cuopt_vars[var_id1] * cuopt_vars[var_id2]
                
                # Constant term
                if hasattr(expr.quadratic, 'constant'):
                    cuopt_expr += expr.quadratic.constant
                    
                return cuopt_expr
                
            else:
                raise ValueError("Unsupported expression type")
                
        except Exception as e:
            logger.error(f"Expression conversion failed: {e}")
            raise
            
    def _solve_with_cuopt(self, model) -> Dict[str, Any]:
        """Solve the cuOpt model"""
        try:
            # Solve the model (parameters already applied)
            model.solve()
            
            # Get solution status
            status = model.get_status()
            
            # Map cuOpt status to OMMX status
            if status == 'optimal':
                ommx_status = "optimal"
            elif status == 'infeasible':
                ommx_status = "infeasible"
            elif status == 'unbounded':
                ommx_status = "unbounded"
            elif status == 'time_limit':
                ommx_status = "time_limit"
            else:
                ommx_status = "unknown"
                
            # Get solution values
            solution = {
                'status': ommx_status,
                'objective_value': model.get_objective_value() if ommx_status == 'optimal' else None,
                'variables': model.get_variable_values() if ommx_status == 'optimal' else {},
                'solve_time': model.get_solve_time()
            }
            
            return solution
            
        except Exception as e:
            logger.error(f"cuOpt solve failed: {e}")
            raise
            
    def _convert_solution_to_ommx(self, solution: Dict[str, Any], ommx_instance: "Instance") -> Dict[str, Any]:
        """Convert cuOpt solution to OMMX format"""
        try:
            if solution['status'] != 'optimal':
                return {
                    "status": solution['status'],
                    "message": f"Problem status: {solution['status']}",
                    "ommx_state_bytes": []
                }
            
            # Create variable ID to value mapping
            var_id_to_value = {}
            var_name_to_id = {var.name: var.id for var in ommx_instance.decision_variables}
            
            for var_name, value in solution['variables'].items():
                if var_name in var_name_to_id:
                    var_id = var_name_to_id[var_name]
                    var_id_to_value[var_id] = float(value)
            
            # Create OMMX State from solution
            logger.info(f"Creating OMMX State with entries: {var_id_to_value}")
            state = ommx.State(var_id_to_value)
            state_bytes = state.to_bytes()
            logger.info(f"OMMX State bytes created: {len(state_bytes)} bytes")
            
            # Create variable name mapping for convenience
            variables = {}
            for var in ommx_instance.decision_variables:
                if var.id in var_id_to_value:
                    variables[var.name] = var_id_to_value[var.id]
            
            return {
                "status": "optimal",
                "objective_value": solution['objective_value'],
                "ommx_state_bytes": list(state_bytes),
                "variables": variables,
                "solve_time": solution['solve_time'],
                "solver": self.name
            }
            
        except Exception as e:
            logger.error(f"Solution conversion failed: {e}")
            raise 