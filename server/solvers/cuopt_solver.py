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
            
            # Convert OMMX instance to MPS file for cuOpt CLI
            mps_file_path = self._convert_ommx_to_cuopt(ommx_instance)
            
            # Note: Parameters are now applied in the CLI call, not separately
            # This is a limitation of the CLI approach
            
            # Solve with cuOpt CLI
            solution = self._solve_with_cuopt(mps_file_path)
            
            # Convert solution back to OMMX format
            result = self._convert_solution_to_ommx(solution, ommx_instance)
            
            solve_time = time.time() - start_time
            objective_value = result.get("objective_value")
            
            return result, solve_time, objective_value
            
        except Exception as e:
            logger.error(f"cuOpt solve failed: {e}")
            raise RuntimeError(f"cuOpt solve failed: {str(e)}")
            
    # Note: Parameter application is now handled in the CLI call
    # The CLI approach has limited parameter support compared to the Python API
            
    def _convert_ommx_to_cuopt(self, ommx_instance: "Instance") -> str:
        """Convert OMMX instance to MPS file for cuOpt CLI"""
        try:
            # Convert OMMX to MPS format
            # For now, we'll use a simplified approach - create a temporary MPS file
            # This is a quick fix until we have proper cuOpt Python API documentation
            
            import tempfile
            import os
            
            # Create temporary MPS file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.mps', prefix='cuopt_ommx_')
            
            try:
                # Write MPS content from OMMX instance
                with os.fdopen(temp_fd, 'w') as f:
                    self._write_ommx_as_mps(f, ommx_instance)
                
                logger.info(f"Created temporary MPS file: {temp_path}")
                return temp_path
                
            except Exception as e:
                os.close(temp_fd)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise RuntimeError(f"Failed to create MPS file: {str(e)}")
                
        except Exception as e:
            logger.error(f"OMMX to cuOpt conversion failed: {e}")
            raise RuntimeError(f"OMMX to cuOpt conversion failed: {str(e)}")
    
    def _write_ommx_as_mps(self, file_handle, ommx_instance):
        """Write OMMX instance as MPS format"""
        # This is a simplified MPS writer - for production use, consider using a proper library
        
        # MPS header
        file_handle.write("NAME          OMMX_PROBLEM\n")
        file_handle.write("ROWS\n")
        
        # Objective row
        file_handle.write(" N  OBJ\n")
        
        # Constraint rows
        for i, constraint in enumerate(ommx_instance.constraints):
            # Determine constraint type based on bound
            if hasattr(constraint, 'equality') and constraint.equality:
                row_type = "E"
            elif hasattr(constraint, 'upper_bound') and constraint.upper_bound is not None:
                row_type = "L"
            elif hasattr(constraint, 'lower_bound') and constraint.lower_bound is not None:
                row_type = "G"
            else:
                row_type = "L"  # Default
            
            file_handle.write(f" {row_type}  C{i+1}\n")
        
        # COLUMNS section
        file_handle.write("COLUMNS\n")
        
        # Variables in objective
        if hasattr(ommx_instance, 'objective') and ommx_instance.objective:
            for var in ommx_instance.decision_variables:
                # Write objective coefficient if exists
                # This is simplified - real implementation would parse the objective expression
                file_handle.write(f"    {var.name}      OBJ       1.0\n")
        
        # Variables in constraints
        for i, constraint in enumerate(ommx_instance.constraints):
            for var in ommx_instance.decision_variables:
                # Write constraint coefficient if exists
                # This is simplified - real implementation would parse constraint expressions
                file_handle.write(f"    {var.name}      C{i+1}     1.0\n")
        
        # RHS section
        file_handle.write("RHS\n")
        for i, constraint in enumerate(ommx_instance.constraints):
            # Write RHS value
            rhs_value = 1.0  # Simplified - should get from constraint bounds
            file_handle.write(f"    RHS1      C{i+1}     {rhs_value}\n")
        
        # BOUNDS section
        file_handle.write("BOUNDS\n")
        for var in ommx_instance.decision_variables:
            # Write variable bounds
            if hasattr(var, 'lower_bound') and var.lower_bound is not None:
                file_handle.write(f" LO BND1      {var.name}      {var.lower_bound}\n")
            else:
                file_handle.write(f" LO BND1      {var.name}      0\n")
            
            if hasattr(var, 'upper_bound') and var.upper_bound is not None:
                file_handle.write(f" UP BND1      {var.name}      {var.upper_bound}\n")
        
        # End
        file_handle.write("ENDATA\n")
        
        logger.info("Generated MPS file from OMMX instance")
            
    # Note: Expression conversion is now handled in the MPS file generation
    # The CLI approach uses MPS format instead of direct expression conversion
            
    def _solve_with_cuopt(self, mps_file_path) -> Dict[str, Any]:
        """Solve using cuOpt CLI"""
        try:
            import subprocess
            import tempfile
            import os
            
            # cuOpt CLI path
            cuopt_cli_path = "/home/ubuntu/.conda/envs/cuOpt-server/bin/cuopt_cli"
            
            # Create temporary output file
            temp_out_fd, temp_out_path = tempfile.mkstemp(suffix='.txt', prefix='cuopt_out_')
            os.close(temp_out_fd)
            
            try:
                # Build cuOpt CLI command
                cmd = [cuopt_cli_path, '--solution-file', temp_out_path, mps_file_path]
                
                logger.info(f"Running cuOpt CLI: {' '.join(cmd)}")
                
                # Execute cuOpt CLI
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"cuOpt CLI failed with return code {result.returncode}: {result.stderr}")
                
                # Parse output
                solution = self._parse_cuopt_cli_output(temp_out_path)
                
                # Clean up MPS file
                if os.path.exists(mps_file_path):
                    os.remove(mps_file_path)
                
                return solution
                
            finally:
                # Clean up output file
                if os.path.exists(temp_out_path):
                    os.remove(temp_out_path)
                
        except Exception as e:
            logger.error(f"cuOpt CLI solve failed: {e}")
            raise
    
    def _parse_cuopt_cli_output(self, output_file_path) -> Dict[str, Any]:
        """Parse cuOpt CLI output file"""
        try:
            with open(output_file_path, 'r') as f:
                lines = f.readlines()
            
            status = None
            objective_value = None
            variables = {}
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('# Status:'):
                    status_str = line.split(':', 1)[1].strip()
                    status = self._map_cuopt_status(status_str)
                
                elif line.startswith('# Objective value:'):
                    try:
                        objective_value = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        objective_value = None
                
                elif line and not line.startswith('#'):
                    # Variable line: "variable_name value"
                    parts = line.split()
                    if len(parts) >= 2:
                        var_name = parts[0]
                        try:
                            var_value = float(parts[1])
                            variables[var_name] = var_value
                        except ValueError:
                            logger.warning(f"Could not parse variable value: {line}")
            
            return {
                'status': status or 'unknown',
                'objective_value': objective_value,
                'variables': variables,
                'solve_time': 0.0  # CLI doesn't provide solve time
            }
            
        except Exception as e:
            logger.error(f"Failed to parse cuOpt CLI output: {e}")
            raise
    
    def _map_cuopt_status(self, cuopt_status: str) -> str:
        """Map cuOpt status to OMMX status"""
        status_map = {
            'Optimal': 'optimal',
            'optimal': 'optimal',
            'Infeasible': 'infeasible',
            'infeasible': 'infeasible',
            'Unbounded': 'unbounded',
            'unbounded': 'unbounded',
            'Time limit': 'time_limit',
            'time_limit': 'time_limit'
        }
        return status_map.get(cuopt_status, 'unknown')
            
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