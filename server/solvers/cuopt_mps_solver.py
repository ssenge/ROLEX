"""
ROLEX Server - cuOpt MPS Solver
MPS file solving using cuOpt CLI
"""
import time
import logging
import subprocess
import os
import tempfile
import uuid
from typing import Dict, Any, Optional

from .mps_base import BaseMPSSolver
from models import MPSOptimizationResponse, SolverDiagnostics
from .mps_parser_utils import get_mps_dimensions

logger = logging.getLogger(__name__)

# cuOpt CLI path - available globally in rolex-server environment
CUOPT_CLI_PATH = "cuopt_cli"


class CuOptMPSSolver(BaseMPSSolver):
    """cuOpt solver with CLI-based MPS file support"""
    
    def __init__(self):
        super().__init__("cuOpt-MPS")
        
    def is_available(self) -> bool:
        """Check if cuOpt CLI is available"""
        return self._check_cuopt_cli()
    
    def _check_cuopt_cli(self) -> bool:
        """Check if cuOpt CLI is available and executable"""
        try:
            # Test actual execution (cuopt_cli is a command in PATH, not a file path)
            result = subprocess.run(
                [CUOPT_CLI_PATH, '--help'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                return True
            else:
                logger.warning(f"cuOpt CLI failed with return code {result.returncode}: {result.stderr}")
                return False
            
        except FileNotFoundError:
            logger.warning(f"cuOpt CLI command '{CUOPT_CLI_PATH}' not found in PATH")
            return False
        except Exception as e:
            logger.error(f"cuOpt CLI check failed: {str(e)}")
            return False
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get cuOpt solver information"""
        info = {
            "name": "cuOpt with CLI-based MPS Support",
            "available": self.is_available(),
            "capabilities": ["linear", "mixed-integer", "mps", "gpu-accelerated"]
        }
        
        if not self.is_available():
            info["status"] = f"cuOpt CLI not available at {CUOPT_CLI_PATH}"
        else:
            try:
                # Try to get version info
                result = subprocess.run(
                    [CUOPT_CLI_PATH, '--help'], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                info["status"] = "available"
                info["cli_path"] = CUOPT_CLI_PATH
            except Exception as e:
                info["status"] = f"error: {str(e)}"
        
        return info
    
    def solve_mps(self, mps_file_path: str, parameters: Dict[str, Any]) -> MPSOptimizationResponse:
        """
        Solve MPS file using cuOpt CLI
        
        Args:
            mps_file_path: Path to MPS file
            parameters: Solver parameters
            
        Returns:
            MPSOptimizationResponse with solution
        """
        if not self.is_available():
            raise RuntimeError("cuOpt CLI is not available")
        
        # Validate parameters
        validated_params = self._validate_parameters(parameters)

        # cuOpt CLI does not support --log-frequency, so remove it if present
        # The log file is created based on its presence, but the CLI itself doesn't take it as an argument.
        cuopt_cli_params = validated_params.copy()
        if 'log_frequency' in cuopt_cli_params:
            del cuopt_cli_params['log_frequency']

        # Get dimensions from MPS file
        mps_parse_start_time = time.time()
        num_variables, num_constraints = get_mps_dimensions(mps_file_path)
        mps_parse_end_time = time.time()
        logger.info(f"MPS parsing (pysmps) took {mps_parse_end_time - mps_parse_start_time:.4f} seconds for {os.path.basename(mps_file_path)}")

        if num_variables is None or num_constraints is None:
            logger.warning(f"Could not determine MPS dimensions for {mps_file_path}. Setting to None.")
            # Proceed with None, as the solver might still work

        # Create temporary output and log files
        temp_output = self._create_temp_output_file()
        temp_log = self._create_temp_output_file('.log') if 'log_frequency' in validated_params else None
        
        try:
            # Build cuOpt CLI command
            cmd = self._build_cuopt_command(mps_file_path, temp_output, cuopt_cli_params, log_file=temp_log)
            
            # Execute cuOpt CLI
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=validated_params.get('max_time', 300) + 30  # Add buffer
            )
            solve_time = time.time() - start_time
            
            # Check for errors
            if result.returncode != 0:
                raise RuntimeError(f"cuOpt CLI failed with return code {result.returncode}: {result.stderr}")
            
            # Parse log file for convergence data
            convergence_data = []
            if temp_log:
                convergence_data = self._parse_cuopt_log(temp_log)

            # Parse output
            response = self._parse_cuopt_output(temp_output, solve_time, validated_params, num_variables, num_constraints)
            response.convergence_data = convergence_data
            return response
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("cuOpt CLI execution timed out")
        except Exception as e:
            logger.error(f"cuOpt CLI execution failed: {str(e)}")
            raise RuntimeError(f"cuOpt CLI execution failed: {str(e)}")
        finally:
            # Clean up temporary files
            self.cleanup_temp_file(temp_output)
            if temp_log:
                self.cleanup_temp_file(temp_log)

    def _parse_cuopt_log(self, log_file: str) -> list:
        """Parse cuOpt log file for convergence data."""
        convergence_data = []
        if not os.path.exists(log_file):
            return convergence_data
        
        with open(log_file, 'r') as f:
            for line in f:
                if 'objective' in line:
                    try:
                        parts = line.split(',')
                        time_part = [p for p in parts if 'time' in p][0]
                        obj_part = [p for p in parts if 'objective' in p][0]
                        
                        time_val = float(time_part.split(':')[1])
                        obj_val = float(obj_part.split(':')[1])
                        
                        convergence_data.append({'time': time_val, 'objective': obj_val})
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse log line: {line} - {e}")
        return convergence_data
    
    def _create_temp_output_file(self, suffix='.txt') -> str:
        """Create temporary output file for cuOpt CLI results"""
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix='cuopt_output_')
        os.close(temp_fd)  # Close the file descriptor, we just need the path
        return temp_path
    
    def _build_cuopt_command(self, mps_file_path: str, output_file: str, parameters: Dict[str, Any], log_file: Optional[str] = None) -> list:
        """Build cuOpt CLI command with parameters"""
        cmd = [CUOPT_CLI_PATH]
        
        # Add time limit if specified
        if 'max_time' in parameters:
            cmd.extend(['--time-limit', str(int(parameters['max_time']))])
        
        # Add solution output file
        cmd.extend(['--solution-file', output_file])

        # Add log file if specified
        if log_file:
            cmd.extend(['--log-file', log_file])
        
        # Add MPS file path
        cmd.append(mps_file_path)
        
        logger.info(f"cuOpt CLI command: {' '.join(cmd)}")
        return cmd
    
    def _parse_cuopt_output(self, output_file: str, solve_time: float, parameters: Dict[str, Any], num_variables: Optional[int], num_constraints: Optional[int]) -> MPSOptimizationResponse:
        """Parse cuOpt CLI output file"""
        
        if not os.path.exists(output_file):
            raise RuntimeError("cuOpt CLI output file not created")
        
        try:
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            status = None
            objective_value = None
            variables = {}
            
            # Parse output lines
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
            
            # Default status if not found
            if status is None:
                status = "unknown"
            
            # Generate message
            message = f"cuOpt CLI status: {status}"
            if objective_value is not None:
                message += f", objective: {objective_value}"
            
            # Create solver info (cuOpt CLI provides limited diagnostics)
            solver_info = SolverDiagnostics()
            # cuOpt CLI doesn't provide iteration/node counts, so leave as None
            
            return MPSOptimizationResponse(
                status=status,
                objective_value=objective_value,
                variables=variables,
                solve_time=solve_time,
                solver=self.name,
                message=message,
                solver_info=solver_info,
                num_variables=num_variables,
                num_constraints=num_constraints
            )
            
        except Exception as e:
            logger.error(f"Failed to parse cuOpt output: {str(e)}")
            raise RuntimeError(f"Failed to parse cuOpt output: {str(e)}")
    
    def _map_cuopt_status(self, cuopt_status: str) -> str:
        """Map cuOpt status to standard status"""
        status_map = {
            'Optimal': 'optimal',
            'optimal': 'optimal',
            'Infeasible': 'infeasible',
            'infeasible': 'infeasible',
            'Unbounded': 'unbounded',
            'unbounded': 'unbounded',
            'Time limit': 'time_limit',
            'time_limit': 'time_limit',
            'Error': 'failed',
            'error': 'failed'
        }
        
        return status_map.get(cuopt_status, 'unknown')
    
    def _validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cuOpt-specific parameters"""
        validated = super()._validate_parameters(parameters)
        
        # cuOpt CLI has limited parameter support
        # For now, we mainly support max_time
        
        # Ensure max_time is reasonable (cuOpt CLI expects integer seconds)
        if 'max_time' in validated:
            validated['max_time'] = max(1, int(validated['max_time']))

        return validated 