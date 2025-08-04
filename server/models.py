"""
ROLEX Server - API Models
Clean request/response models for MPS optimization jobs
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from enum import Enum

# Version configuration
SERVER_NAME = "ROLEX"
SERVER_VERSION = "0.0.1"
SERVER_DESCRIPTION = "Remote Optimization Library EXecution"


class JobStatus(str, Enum):
    """Job execution status"""
    QUEUED = "queued"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    TIMELIMIT_REACHED = "timelimit_reached"


class MPSSolverType(str, Enum):
    """Available MPS solver types"""
    GUROBI = "gurobi"
    CUOPT = "cuopt"
    GLOP = "glop"
    PYCUOPT = "pycuopt"


class MPSOptimizationRequest(BaseModel):
    """Request to solve an MPS optimization problem"""
    solver: MPSSolverType
    parameters: Optional[Dict[str, Any]] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "solver": "gurobi",
                "parameters": {
                    "max_time": 300,
                    "gap_tolerance": 0.01,
                    "threads": 4,
                    "verbose": True
                }
            }
        }


class SolverDiagnostics(BaseModel):
    """Solver diagnostic information"""
    iterations: Optional[int] = None
    nodes: Optional[int] = None
    gap: Optional[float] = None
    bound: Optional[float] = None
    status_code: Optional[int] = None


class ConvergencePoint(BaseModel):
    """Data point for convergence analysis"""
    time: float
    objective: float


class MPSOptimizationResponse(BaseModel):
    """Response containing MPS optimization results"""
    status: str
    objective_value: Optional[float] = None
    variables: Dict[str, float] = {}
    solve_time: Optional[float] = None
    total_time: Optional[float] = None
    num_constraints: Optional[int] = None
    solver: Optional[str] = None
    message: Optional[str] = None
    parameters_used: Dict[str, Any] = {}
    solver_info: SolverDiagnostics = SolverDiagnostics()
    convergence_data: Optional[List[ConvergencePoint]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "status": "optimal",
                "objective_value": 42.0,
                "variables": {
                    "x1": 1.0,
                    "x2": 2.0
                },
                "solve_time": 0.123,
                "total_time": 0.456,
                "solver": "gurobi",
                "message": "Optimization terminated successfully",
                "parameters_used": {
                    "max_time": 300,
                    "gap_tolerance": 0.01
                }
            }
        }


class JobSubmissionResponse(BaseModel):
    """Response when submitting a job"""
    job_id: str
    status: str
    message: str


class MPSJobStatusResponse(BaseModel):
    """Response when checking MPS job status"""
    job_id: str
    status: str
    submitted_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[MPSOptimizationResponse] = None
    error: Optional[str] = None


class SolverInfo(BaseModel):
    """Information about a solver"""
    name: str
    available: bool
    version: Optional[str] = None
    license: Optional[str] = None
    capabilities: List[str] = []


class ServerInfo(BaseModel):
    """Server status and information"""
    name: str = SERVER_NAME
    version: str = SERVER_VERSION
    status: str = "running"
    available_solvers: Dict[str, SolverInfo]
    active_jobs: int
    total_jobs: int 