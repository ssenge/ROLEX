"""
ROLEX Server - pyCuOpt MPS Solver
MPS file solving using the cuOpt Python API with a dual-strategy for LP and MIP problems.
"""
import time
import logging
import os
import tempfile
from typing import Dict, Any, Optional, List

from .mps_base import BaseMPSSolver
from models import MPSOptimizationResponse, SolverDiagnostics, ConvergencePoint

# Import dependencies with proper error handling
try:
    from cuopt.linear_programming import solver, solver_settings
    from cuopt.linear_programming.internals import GetSolutionCallback
    from cuopt.linear_programming.solver.solver_parameters import CUOPT_TIME_LIMIT, CUOPT_LOG_FILE
    import cuopt_mps_parser
    CUOPT_AVAILABLE = True
except ImportError:
    CUOPT_AVAILABLE = False

logger = logging.getLogger(__name__)


class PyCuOptMPSSolver(BaseMPSSolver):
    """cuOpt solver using the native Python API"""

    def __init__(self):
        super().__init__("pyCuOpt")

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
                # --- LP Strategy: Use Log File Parsing ---
                temp_fd, temp_log_path = tempfile.mkstemp(suffix='.log', prefix='cuopt_lp_log_')
                os.close(temp_fd)
                settings.set_parameter(CUOPT_LOG_FILE, temp_log_path)

            # Solve the problem
            solve_start_time = time.time()
            solution = solver.Solve(data_model, settings)
            solve_time = time.time() - solve_start_time

            # If it was an LP, parse the log file now
            if temp_log_path:
                lp_objectives = self._parse_lp_log(temp_log_path)
                # For LPs, we don't have precise timestamps, so time is 0
                convergence_data = [ConvergencePoint(time=0, objective=obj) for obj in lp_objectives]

            # Process the results
            response = self._process_cuopt_results(solution, solve_time, data_model, validated_params)
            response.convergence_data = convergence_data
            return response

        except Exception as e:
            logger.error(f"cuOpt Python API solve failed: {str(e)}")
            raise RuntimeError(f"cuOpt solve failed: {str(e)}")
        finally:
            if temp_log_path and os.path.exists(temp_log_path):
                os.remove(temp_log_path)

    def _parse_lp_log(self, log_path: str) -> List[float]:
        """Best-effort parsing of a cuOpt LP log file for objective values."""
        objectives = []
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    # This is a heuristic based on observed log formats.
                    # It looks for lines from the PDLP solver's progress table.
                    if 'primal_objective' in line and 'dual_objective' in line:
                        try:
                            parts = line.split()
                            # Find the index for primal_objective and extract the value
                            obj_index = parts.index('primal_objective') + 2 # The value is usually 2 positions after the label
                            if obj_index < len(parts):
                                objectives.append(float(parts[obj_index]))
                        except (ValueError, IndexError):
                            continue # Ignore lines that don't match the expected format
        except Exception as e:
            logger.warning(f"Could not parse LP log file {log_path}: {e}")
        return objectives

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
