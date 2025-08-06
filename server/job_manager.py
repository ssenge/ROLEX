"""
ROLEX Server - Job Manager
Business logic layer for managing MPS optimization jobs
"""
import uuid
import time
import threading
import os
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

from models import JobStatus, MPSSolverType, MPSOptimizationResponse, MPSJobStatusResponse, MPSOptimizationRequest
from solvers.gurobi_mps_solver import GurobiMPSSolver
from solvers.cuopt_mps_solver import CuOptMPSSolver
from solvers.pycuopt_mps_solver import PyCuOptMPSSolver
from solvers.ortools_mps_solver import ORToolsGLOPSolver, ORToolsCBCSolver, ORToolsCLPSolver, ORToolsSCIPSolver, ORToolsPDLPSolver
from solvers.scipy_mps_solver import SciPyMPSSolver
from solvers.pyomo_mps_solver import (
    PyomoCPLEXSolver, PyomoGurobiSolver, PyomoGLPKSolver,
    PyomoCBCSolver, PyomoIPOPTSolver, PyomoSCIPSolver, PyomoHiGHSSolver
)


logger = logging.getLogger(__name__)


class MPSJob:
    """Represents an MPS optimization job"""
    
    def __init__(self, job_id: str, solver_type: MPSSolverType, mps_file_path: str, parameters: Dict[str, Any], optimality_tolerance: Optional[float] = None):
        self.job_id = job_id
        self.solver_type = solver_type
        self.mps_file_path = mps_file_path
        self.parameters = parameters
        self.optimality_tolerance = optimality_tolerance
        self.status = JobStatus.QUEUED
        self.submitted_at = datetime.now().isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.result: Optional[MPSOptimizationResponse] = None
        self.error: Optional[str] = None
        self.temp_file_cleanup = True  # Flag to clean up temp files
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for API response"""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result.model_dump() if self.result else None,
            "error": self.error
        }


class JobManager:
    """Manages MPS optimization jobs and solver instances"""
    
    def __init__(self, max_workers: int = 1):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.mps_jobs: Dict[str, MPSJob] = {}
        self.mps_solvers: Dict[MPSSolverType, Any] = {}
        self.lock = threading.Lock()
        self.total_jobs = 0
        
        # Initialize MPS solvers
        self._init_mps_solvers()
        
        # Log initialization
        logger.info(f"JobManager initialized with {max_workers} workers")
        self._log_solver_status()
    
    def _init_mps_solvers(self):
        """Initialize MPS solvers"""
        # Gurobi MPS solver
        try:
            self.mps_solvers[MPSSolverType.GUROBI] = GurobiMPSSolver()
            logger.info("Gurobi MPS solver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gurobi MPS solver: {e}")
        
        # cuOpt MPS solver
        try:
            self.mps_solvers[MPSSolverType.CUOPT] = CuOptMPSSolver()
            logger.info("cuOpt MPS solver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize cuOpt MPS solver: {e}")

        # pyCuOpt MPS solver
        try:
            self.mps_solvers[MPSSolverType.PYCUOPT] = PyCuOptMPSSolver()
            logger.info("pyCuOpt MPS solver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize pyCuOpt MPS solver: {e}")

        # OR-Tools GLOP solver
        try:
            self.mps_solvers[MPSSolverType.ORTOOLS_GLOP] = ORToolsGLOPSolver()
            logger.info("OR-Tools GLOP solver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OR-Tools GLOP solver: {e}")

        # OR-Tools CBC solver
        try:
            self.mps_solvers[MPSSolverType.ORTOOLS_CBC] = ORToolsCBCSolver()
            logger.info("OR-Tools CBC solver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OR-Tools CBC solver: {e}")

        # OR-Tools CLP solver
        try:
            self.mps_solvers[MPSSolverType.ORTOOLS_CLP] = ORToolsCLPSolver()
            logger.info("OR-Tools CLP solver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OR-Tools CLP solver: {e}")

        # OR-Tools SCIP solver
        try:
            self.mps_solvers[MPSSolverType.ORTOOLS_SCIP] = ORToolsSCIPSolver()
            logger.info("OR-Tools SCIP solver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OR-Tools SCIP solver: {e}")

        # OR-Tools PDLP solver
        try:
            self.mps_solvers[MPSSolverType.ORTOOLS_PDLP] = ORToolsPDLPSolver()
            logger.info("OR-Tools PDLP solver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OR-Tools PDLP solver: {e}")

        # SciPy LP solver
        try:
            self.mps_solvers[MPSSolverType.SCIPY_LP] = SciPyMPSSolver()
            logger.info("SciPy LP solver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SciPy LP solver: {e}")
        
        # Initialize Pyomo solvers
        try:
            self.mps_solvers[MPSSolverType.PYOMO_CPLEX] = PyomoCPLEXSolver()
            logger.info("Pyomo CPLEX solver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Pyomo CPLEX solver: {e}")
            
        try:
            self.mps_solvers[MPSSolverType.PYOMO_GUROBI] = PyomoGurobiSolver()
            logger.info("Pyomo Gurobi solver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Pyomo Gurobi solver: {e}")
            
        try:
            self.mps_solvers[MPSSolverType.PYOMO_GLPK] = PyomoGLPKSolver()
            logger.info("Pyomo GLPK solver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Pyomo GLPK solver: {e}")
            
        try:
            self.mps_solvers[MPSSolverType.PYOMO_CBC] = PyomoCBCSolver()
            logger.info("Pyomo CBC solver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Pyomo CBC solver: {e}")
            
        try:
            self.mps_solvers[MPSSolverType.PYOMO_IPOPT] = PyomoIPOPTSolver()
            logger.info("Pyomo IPOPT solver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Pyomo IPOPT solver: {e}")
            
        try:
            self.mps_solvers[MPSSolverType.PYOMO_SCIP] = PyomoSCIPSolver()
            logger.info("Pyomo SCIP solver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Pyomo SCIP solver: {e}")
            
        try:
            self.mps_solvers[MPSSolverType.PYOMO_HIGHS] = PyomoHiGHSSolver()
            logger.info("Pyomo HiGHS solver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Pyomo HiGHS solver: {e}")
    
    def _log_solver_status(self):
        """Log the availability of all solvers"""
        available_mps = [solver.value for solver in MPSSolverType if self.is_mps_solver_available(solver)]
        logger.info(f"Available MPS solvers: {available_mps}")
    
    def get_available_mps_solvers(self) -> Dict[str, Dict[str, Any]]:
        """Get available MPS solvers and their info"""
        available = {}
        for solver_type in MPSSolverType:
            solver_instance = self.mps_solvers.get(solver_type)
            if solver_instance:
                available[solver_type.value] = solver_instance.get_solver_info()
        return available
    
    def is_mps_solver_available(self, solver_type: MPSSolverType) -> bool:
        """Check if an MPS solver is available"""
        return solver_type in self.mps_solvers and self.mps_solvers[solver_type] is not None
    
    async def submit_mps_job(self, request: MPSOptimizationRequest, mps_file_path: str) -> str:
        """Submit an MPS optimization job"""
        print(f"=== DEBUG: submit_mps_job called with solver {request.solver.value} ===")
        
        if not self.is_mps_solver_available(request.solver):
            print(f"=== DEBUG: Solver {request.solver.value} is not available ===")
            raise ValueError(f"Solver {request.solver.value} is not available")
        
        print(f"=== DEBUG: Solver {request.solver.value} is available ===")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        print(f"=== DEBUG: Generated job ID: {job_id} ===")
        
        # Create job
        job = MPSJob(
            job_id=job_id,
            solver_type=request.solver,
            mps_file_path=mps_file_path,
            parameters=request.parameters,
            optimality_tolerance=request.optimality_tolerance
        )
        print(f"=== DEBUG: Created job object ===")
        
        # Store job
        with self.lock:
            self.mps_jobs[job_id] = job
            self.total_jobs += 1
        print(f"=== DEBUG: Stored job in manager ===")
        
        # Submit to executor
        future = self.executor.submit(self._execute_mps_job, job_id)
        print(f"=== DEBUG: Submitted job to executor ===")
        
        logger.info(f"Submitted MPS job {job_id} with {request.solver.value} solver")
        return job_id
    
    async def get_mps_job_status(self, job_id: str) -> MPSJobStatusResponse:
        """Get the status of an MPS job"""
        with self.lock:
            if job_id not in self.mps_jobs:
                raise KeyError(f"Job {job_id} not found")
            
            job = self.mps_jobs[job_id]
            return MPSJobStatusResponse(
                job_id=job.job_id,
                status=job.status,
                submitted_at=job.submitted_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                result=job.result,
                error=job.error
            )
    
    async def list_jobs(self) -> Dict[str, Any]:
        """List all jobs"""
        with self.lock:
            jobs = []
            for job in self.mps_jobs.values():
                jobs.append({
                    "job_id": job.job_id,
                    "status": job.status,
                    "solver": job.solver_type.value,
                    "submitted_at": job.submitted_at
                })
            
            return {
                "total_jobs": len(jobs),
                "jobs": jobs
            }
    
    def get_active_jobs_count(self) -> int:
        """Get count of active jobs"""
        with self.lock:
            return len([j for j in self.mps_jobs.values() if j.status in [JobStatus.QUEUED, JobStatus.RUNNING]])
    
    def get_total_jobs_count(self) -> int:
        """Get total jobs count"""
        return self.total_jobs
    
    def _execute_mps_job(self, job_id: str) -> None:
        """Execute an MPS job"""
        print(f"=== DEBUG: _execute_mps_job called for job {job_id} ===")
        
        with self.lock:
            if job_id not in self.mps_jobs:
                print(f"=== DEBUG: Job {job_id} not found in jobs dict ===")
                logger.error(f"Job {job_id} not found")
                return
            
            job = self.mps_jobs[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now().isoformat()
        
        print(f"=== DEBUG: Job {job_id} status set to RUNNING ===")
        logger.info(f"Executing MPS job {job_id} with {job.solver_type.value} solver")
        
        try:
            # Get solver instance
            solver = self.mps_solvers[job.solver_type]
            print(f"=== DEBUG: Got solver instance: {solver} ===")
            
            # Execute optimization
            print(f"=== DEBUG: Calling solve_with_timing ===")
            print(f"=== DEBUG: Solver type: {type(solver)} ===")
            print(f"=== DEBUG: Solver MRO: {type(solver).__mro__} ===")
            print(f"=== DEBUG: solve_with_timing method: {solver.solve_with_timing} ===")
            
            import inspect
            print(f"=== DEBUG: solve_with_timing signature: {inspect.signature(solver.solve_with_timing)} ===")
            
            result = solver.solve_with_timing(job.mps_file_path, job.parameters, job.optimality_tolerance)
            print(f"=== DEBUG: solve_with_timing completed with status: {result.status} ===")
            logger.debug(f"Job {job_id} convergence_data: {result.convergence_data}")
            logger.debug(f"Job {job_id} parameters_used: {result.parameters_used}")
            
            # Update job with result
            with self.lock:
                job.result = result
                job.status = result.status
                job.completed_at = datetime.now().isoformat()
            
            print(f"=== DEBUG: Job {job_id} completed successfully ===")
            logger.info(f"Job {job_id} completed with status: {result.status}")
            
        except Exception as e:
            print(f"=== DEBUG: Job {job_id} failed with exception: {str(e)} ===")
            print(f"=== DEBUG: Exception type: {type(e).__name__} ===")
            print(f"=== DEBUG: Traceback: {traceback.format_exc()} ===")
            
            logger.error(f"Job {job_id} failed: {str(e)}")
            
            with self.lock:
                job.error = str(e)
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now().isoformat()
        
        finally:
            # Clean up temporary MPS file
            if job.temp_file_cleanup:
                try:
                    if os.path.exists(job.mps_file_path):
                        os.unlink(job.mps_file_path)
                        print(f"=== DEBUG: Cleaned up temp file: {job.mps_file_path} ===")
                        logger.debug(f"Cleaned up temp file: {job.mps_file_path}")
                except Exception as e:
                    print(f"=== DEBUG: Failed to clean up temp file: {e} ===")
                    logger.warning(f"Failed to clean up temp file {job.mps_file_path}: {e}")
    
    async def shutdown(self):
        """Shutdown the job manager"""
        logger.info("Shutting down JobManager...")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Clean up any remaining temp files
        with self.lock:
            for job in self.mps_jobs.values():
                if job.temp_file_cleanup and os.path.exists(job.mps_file_path):
                    try:
                        os.unlink(job.mps_file_path)
                        logger.debug(f"Cleaned up temp file: {job.mps_file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file {job.mps_file_path}: {e}")
        
        logger.info("JobManager shutdown complete")
    
    def __len__(self) -> int:
        """Return the number of jobs"""
        with self.lock:
            return len(self.mps_jobs)
    
    def __contains__(self, job_id: str) -> bool:
        """Check if job exists"""
        with self.lock:
            return job_id in self.mps_jobs 