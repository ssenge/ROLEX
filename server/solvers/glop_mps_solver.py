"""
ROLEX Server - GLOP MPS Solver
"""

import logging
import os
import time
from typing import Any, Dict

from ortools.linear_solver.python import model_builder

from .mps_base import BaseMPSSolver
from models import MPSOptimizationResponse, SolverDiagnostics

logger = logging.getLogger(__name__)

class GlopMPSSolver(BaseMPSSolver):
    """GLOP MPS Solver implementation"""

    def __init__(self):
        super().__init__("glop")
        self._solver_instance = None
        self._solver_version = "unknown"
        self._check_availability()

    def _check_availability(self):
        """Internal method to check GLOP availability and get version."""
        try:
            # Attempt to create a solver instance to check availability
            solver = model_builder.ModelSolver("GLOP")
            self._available = True
            # OR-Tools does not expose a direct version for GLOP,
            # but we can use the OR-Tools package version if needed.
            # For now, we'll keep it as "unknown" or derive from ortools.__version__
            # if that becomes necessary.
            try:
                import ortools
                self._solver_version = ortools.__version__
            except ImportError:
                self._solver_version = "ortools_version_unknown"
            logger.info(f"GLOP solver initialized. OR-Tools version: {self._solver_version}")
        except Exception as e:
            self._available = False
            logger.warning(f"GLOP solver not available: {e}")

    def is_available(self) -> bool:
        """Check if GLOP solver is available and ready to use."""
        return self._available

    def get_solver_info(self) -> Dict[str, Any]:
        """Get detailed information about the GLOP solver."""
        if self._solver_info is None:
            self._solver_info = {
                "name": self.name,
                "available": self.is_available(),
                "version": self._solver_version,
                "description": "Google's GLOP (Google Linear Optimization Package) solver for Linear Programs.",
                "capabilities": ["LP"],
                "parameters": {
                    "max_time": "Supported (in seconds).",
                    "verbose": "Not supported in this integration."
                }
            }
        return self._solver_info

    def solve_mps(self, mps_file_path: str, parameters: Dict[str, Any]) -> MPSOptimizationResponse:
        """
        Solve MPS optimization problem using GLOP.
        """
        if not self.is_available():
            raise RuntimeError("GLOP solver is not available.")

        validated_params = self._validate_parameters(parameters)

        model = model_builder.ModelBuilder()
        try:
            model.import_from_mps_file(mps_file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to import MPS file into GLOP: {e}")

        solver = model_builder.ModelSolver("GLOP")

        time_limit = validated_params.get('max_time')
        if time_limit and time_limit > 0:
            solver.set_time_limit_in_seconds(time_limit)
            logger.info(f"GLOP solver time limit set to {time_limit} seconds.")

        solve_start_time = time.time()
        status = solver.solve(model)
        solve_end_time = time.time()
        solve_time = solve_end_time - solve_start_time

        objective_value = None
        variables = {}
        solver_status = "unknown"

        if status == model_builder.SolveStatus.OPTIMAL:
            solver_status = "optimal"
            objective_value = solver.objective_value
            for i in range(model.num_variables):
                var = model.var_from_index(i)
                variables[var.name] = solver.value(var)
        elif status == model_builder.SolveStatus.FEASIBLE:
            solver_status = "feasible"
            objective_value = solver.objective_value
            for i in range(model.num_variables):
                var = model.var_from_index(i)
                variables[var.name] = solver.value(var)
        elif status == model_builder.SolveStatus.INFEASIBLE:
            solver_status = "infeasible"
        elif status == model_builder.SolveStatus.UNBOUNDED:
            solver_status = "unbounded"
        elif status == model_builder.SolveStatus.MODEL_INVALID:
            solver_status = "failed"
            logger.error("GLOP: Model is invalid.")
        elif status == model_builder.SolveStatus.NOT_SOLVED:
            # This status can be returned when a time limit is reached before a solution is found.
            if time_limit and solve_time >= time_limit * 0.99:
                solver_status = "timelimit_reached"
                logger.warning("GLOP: Time limit reached, no solution found.")
            else:
                solver_status = "failed"
                logger.warning("GLOP: Problem not solved.")
        elif status == model_builder.SolveStatus.ABNORMAL:
            solver_status = "failed"
            logger.error("GLOP: Abnormal termination.")
        elif status == model_builder.SolveStatus.UNKNOWN_STATUS:
            solver_status = "failed"
            logger.error(f"GLOP: Unknown status.")
        else:
            solver_status = "failed"
            logger.error(f"GLOP: Unexpected solve status: {status}")

        # Check if time limit was reached even if a feasible/optimal solution was found
        if time_limit and solve_time >= time_limit * 0.99 and solver_status in ['optimal', 'feasible']:
            solver_status = "timelimit_reached"
            logger.warning(f"GLOP: Time limit reached, returning best found solution with status '{solver_status}'.")

        return MPSOptimizationResponse(
            status=solver_status,
            objective_value=objective_value,
            variables=variables,
            solve_time=solve_time,
            solver=self.name,
            message=f"GLOP solve completed with status: {solver_status}",
            solver_info=SolverDiagnostics(
                solver_version=self._solver_version,
                solve_status_code=status.value,
                solve_status_message=str(status)
            ),
            num_constraints=model.num_constraints
        )


