"""
ROLEX Server - pyCuOpt MPS Solver
MPS file solving using the cuOpt Python API with a dual-strategy for LP and MIP problems.
"""
import time
import logging
import os
import tempfile
import sys
import io
from typing import Dict, Any, Optional, List

from .mps_base import BaseMPSSolver
from models import MPSOptimizationResponse, SolverDiagnostics, ConvergencePoint

# Import dependencies with proper error handling
try:
    from cuopt.linear_programming import solver, solver_settings
    from cuopt.linear_programming.internals import GetSolutionCallback
    from cuopt.linear_programming.solver.solver_parameters import CUOPT_TIME_LIMIT, CUOPT_LOG_FILE, CUOPT_LOG_TO_CONSOLE
    import cuopt_mps_parser
    CUOPT_AVAILABLE = True
except ImportError:
    CUOPT_AVAILABLE = False

logger = logging.getLogger(__name__)


class PyCuOptMPSSolver(BaseMPSSolver):
    """cuOpt solver using the native Python API"""

    def __init__(self):
        super().__init__("pyCuOpt")
        self.last_lp_log_path = None

    def is_available(self) -> bool:
        return CUOPT_AVAILABLE

    def get_solver_info(self) -> Dict[str, Any]:
        info = {
            "name": "cuOpt with Python API",
            "available": self.is_available(),
            "capabilities": ["linear", "mixed-integer", "mps", "gpu-accelerated"]
        }
        if not CUOPT_AVAILABLE:
            info["status"] = "cuOpt Python libraries not found"
        else:
            info["status"] = "available"
        return info

    def solve_mps(self, mps_file_path: str, parameters: Dict[str, Any]) -> MPSOptimizationResponse:
        if not self.is_available():
            raise RuntimeError("cuOpt Python API is not available")

        validated_params = self._validate_parameters(parameters)
        temp_log_path = None

        try:
            data_model = cuopt_mps_parser.ParseMps(mps_file_path)
            settings = solver_settings.SolverSettings()
            if 'max_time' in validated_params:
                settings.set_parameter(CUOPT_TIME_LIMIT, validated_params['max_time'])

            convergence_data = []
            is_mip = "I" in data_model.variable_types

            if is_mip and 'log_frequency' in validated_params:
                # --- MIP Strategy: Use Incumbent Callback ---
                last_log_time = -float('inf')
                start_time_mip = 0.0

                class CustomGetSolutionCallback(GetSolutionCallback):
                    def get_solution(self, solution, solution_cost):
                        nonlocal last_log_time, start_time_mip
                        current_time = time.time() - start_time_mip
                        if (current_time - last_log_time) >= validated_params['log_frequency']:
                            cost = solution_cost.copy_to_host()[0]
                            convergence_data.append(ConvergencePoint(time=current_time, objective=cost))
                            last_log_time = current_time
                            logger.debug(f"PyCuOpt MIP Callback: current_time={current_time}, cost={cost}, log_frequency={validated_params['log_frequency']}")
                
                settings.set_mip_callback(CustomGetSolutionCallback())
                start_time_mip = time.time()

            elif not is_mip:
                # --- LP Strategy: Redirect Console Log to File ---
                settings.set_parameter(CUOPT_LOG_TO_CONSOLE, True)
                
                temp_fd, self.last_lp_log_path = tempfile.mkstemp(suffix='.log', prefix='cuopt_lp_console_log_')
                os.close(temp_fd)
                
                # Store original stdout/stderr
                _original_stdout = sys.stdout
                _original_stderr = sys.stderr
                
                # Redirect stdout/stderr to the temporary file
                sys.stdout = open(self.last_lp_log_path, 'w')
                sys.stderr = open(self.last_lp_log_path, 'w')

            # Solve the problem
            solve_start_time = time.time()
            solution = solver.Solve(data_model, settings)
            solve_time = time.time() - solve_start_time

            # For LPs, intermediate convergence data will be in the redirected log file.
            # For MIPs, convergence_data is populated by the callback.

            # Process the results
            response = self._process_cuopt_results(solution, solve_time, data_model, validated_params)
            response.convergence_data = convergence_data
            return response

        except Exception as e:
            logger.error(f"cuOpt Python API solve failed: {str(e)}")
            raise RuntimeError(f"cuOpt solve failed: {str(e)}")
        finally:
            # Ensure stdout/stderr are restored
            if not is_mip and '_original_stdout' in locals():
                if sys.stdout != _original_stdout: # Check if it was redirected
                    sys.stdout.close()
                if sys.stderr != _original_stderr: # Check if it was redirected
                    sys.stderr.close()
                sys.stdout = _original_stdout
                sys.stderr = _original_stderr
            # The temporary log file is NOT removed here, so the user can inspect it.

    def _process_cuopt_results(self, solution, solve_time: float, data_model, parameters: Dict[str, Any]) -> MPSOptimizationResponse:
        status_map = { 0: "optimal", 1: "optimal", 2: "time_limit", 3: "interrupted", -1: "infeasible", -2: "unbounded", -3: "failed" }
        raw_status = solution.get_termination_status()
        status = status_map.get(raw_status, "unknown")
        objective_value = solution.get_primal_objective()
        variables = solution.get_vars()
        solver_info = SolverDiagnostics()
        message = f"cuOpt status: {status}, objective: {objective_value}"

        return MPSOptimizationResponse(
            status=status,
            objective_value=objective_value,
            variables=variables,
            solve_time=solve_time,
            solver=self.name,
            message=message,
            solver_info=solver_info,
            num_constraints=len(data_model.b),
            num_variables=len(data_model.c),
            parameters_used=parameters
        )

    def _validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        validated = super()._validate_parameters(parameters)
        if 'log_frequency' in parameters:
            freq = parameters['log_frequency']
            if isinstance(freq, (int, float)) and freq > 0:
                validated['log_frequency'] = float(freq)
        return validated
