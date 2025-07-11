"""
ROLEX Server - Job Manager
Business logic layer for managing solvers and jobs
"""
import uuid
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

from models import JobStatus, SolverType, OptimizationResponse, JobStatusResponse
from solvers.gurobi_solver import GurobiSolver
from solvers.cuopt_solver import CuOptSolver
from solvers.scipy_solver import SciPyLPSolver


logger = logging.getLogger(__name__)


class Job:
    """Represents an optimization job"""
    
    def __init__(self, job_id: str, solver_type: SolverType, model: Dict[str, Any], parameters: Dict[str, Any]):
        self.job_id = job_id
        self.solver_type = solver_type
        self.model = model
        self.parameters = parameters
        self.status = JobStatus.QUEUED
        self.submitted_at = datetime.now().isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.result: Optional[OptimizationResponse] = None
        self.error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for API response"""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result.dict() if self.result else None,
            "error": self.error
        }


class JobManager:
    """Manages available solvers and job execution"""
    
    def __init__(self, max_workers: int = 1):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Job storage (in production, use a database)
        self.jobs: Dict[str, Job] = {}
        self.jobs_lock = threading.Lock()
        
        # Initialize available solvers
        self.solvers = {
            SolverType.GUROBI: GurobiSolver(),
            SolverType.CUOPT: CuOptSolver(),
            SolverType.SCIPY: SciPyLPSolver(),  # Basic fallback solver
            # Add more solvers here as they become available
        }
        
        # Statistics
        self.total_jobs = 0
        self.active_jobs = 0
        
        logger.info(f"JobManager initialized with {max_workers} concurrent solver instances")
        self._log_solver_status()
    
    def _log_solver_status(self):
        """Log the status of all solvers"""
        for solver_type, solver in self.solvers.items():
            status = "available" if solver.is_available() else "unavailable"
            logger.info(f"Solver {solver_type}: {status}")
    
    def get_available_solvers(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available solvers"""
        result = {}
        for solver_type, solver in self.solvers.items():
            result[solver_type.value] = solver.get_solver_info()
        return result
    
    def is_solver_available(self, solver_type: SolverType) -> bool:
        """Check if a specific solver is available"""
        solver = self.solvers.get(solver_type)
        return solver is not None and solver.is_available()
    
    def submit_job(self, solver_type: SolverType, model: Dict[str, Any], parameters: Dict[str, Any]) -> str:
        """
        Submit an optimization job
        
        Args:
            solver_type: Which solver to use
            model: OMMX model as dictionary
            parameters: Solver parameters
            
        Returns:
            Job ID string
            
        Raises:
            ValueError: If solver is not available or model is invalid
        """
        # Validate solver availability
        if not self.is_solver_available(solver_type):
            available_solvers = [st.value for st, s in self.solvers.items() if s.is_available()]
            raise ValueError(f"Solver {solver_type.value} is not available. Available solvers: {available_solvers}")
        
        # Model validation is handled by the solver itself
        
        # Create job
        job_id = str(uuid.uuid4())
        job = Job(job_id, solver_type, model, parameters)
        
        # Store job
        with self.jobs_lock:
            self.jobs[job_id] = job
            self.total_jobs += 1
            self.active_jobs += 1
        
        # Submit for execution
        future = self.executor.submit(self._execute_job, job_id)
        
        logger.info(f"Job {job_id} submitted for {solver_type.value} solver")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[JobStatusResponse]:
        """Get the status of a job"""
        with self.jobs_lock:
            job = self.jobs.get(job_id)
        
        if job is None:
            return None
        
        return JobStatusResponse(**job.to_dict())
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        return {
            "available_solvers": self.get_available_solvers(),
            "active_jobs": self.active_jobs,
            "total_jobs": self.total_jobs,
            "max_workers": self.max_workers
        }
    
    def _execute_job(self, job_id: str) -> None:
        """Execute a job (runs in thread pool)"""
        with self.jobs_lock:
            job = self.jobs.get(job_id)
        
        if job is None:
            logger.error(f"Job {job_id} not found during execution")
            return
        
        try:
            # Update job status
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now().isoformat()
            
            logger.info(f"Starting job {job_id} with {job.solver_type.value} solver")
            
            # Get solver and execute
            solver = self.solvers[job.solver_type]
            result_dict = solver.solve_with_timing(job.model, job.parameters)
            
            # Create response
            result = OptimizationResponse(**result_dict)
            
            # Update job with results
            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now().isoformat()
            
            logger.info(f"Job {job_id} completed successfully with status: {result.status}")
            
        except Exception as e:
            # Handle execution errors
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now().isoformat()
            
            logger.error(f"Job {job_id} failed: {str(e)}")
        
        finally:
            # Decrement active job count
            with self.jobs_lock:
                self.active_jobs -= 1
    
    def shutdown(self):
        """Gracefully shutdown the job manager"""
        logger.info("Shutting down JobManager...")
        self.executor.shutdown(wait=True)
        logger.info("JobManager shutdown complete")
    
    def __len__(self) -> int:
        """Return number of jobs"""
        return len(self.jobs)
    
    def __contains__(self, job_id: str) -> bool:
        """Check if job exists"""
        with self.jobs_lock:
            return job_id in self.jobs 