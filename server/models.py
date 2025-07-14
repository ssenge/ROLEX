"""
ROLEX Server - API Models
Clean request/response models for optimization jobs
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


class SolverType(str, Enum):
    """Available solver types"""
    GUROBI = "gurobi"
    CPLEX = "cplex"  # Future extension
    CUOPT = "cuopt"  # GPU-accelerated solver
    SCIPY = "scipy"  # Basic fallback solver


class MPSSolverType(str, Enum):
    """Available MPS solver types"""
    GUROBI = "gurobi"
    CUOPT = "cuopt"


class OptimizationRequest(BaseModel):
    """Request to solve an optimization problem"""
    solver: SolverType
    model: List[int]  # OMMX model as bytes (list of integers)
    parameters: Optional[Dict[str, Any]] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "solver": "gurobi",
                "model": {
                    "variables": [
                        {"id": 1, "name": "x1", "type": "continuous", "bound": {"lower": 0, "upper": 10}},
                        {"id": 2, "name": "x2", "type": "continuous", "bound": {"lower": 0, "upper": 10}}
                    ],
                    "constraints": [
                        {"id": 1, "name": "sum_constraint", "expression": "x1 + x2 <= 5"}
                    ],
                    "objective": {
                        "sense": "maximize",
                        "expression": "3*x1 + 2*x2"
                    }
                },
                "parameters": {
                    "time_limit": 60,
                    "verbose": True
                }
            }
        }


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
                    "threads": 4,
                    "gap_tolerance": 0.01,
                    "verbose": True
                }
            }
        }


class OptimizationResponse(BaseModel):
    """Response containing optimization results"""
    status: str
    objective_value: Optional[float] = None
    variables: Dict[str, float] = {}
    solve_time: Optional[float] = None
    solver: Optional[str] = None
    message: Optional[str] = None
    ommx_state_bytes: Optional[List[int]] = None  # OMMX State as bytes (list of integers)


class SolverDiagnostics(BaseModel):
    """Solver diagnostic information"""
    iterations: Optional[int] = None
    nodes: Optional[int] = None
    gap: Optional[float] = None
    bound: Optional[float] = None
    status_code: Optional[int] = None


class MPSOptimizationResponse(BaseModel):
    """Response containing MPS optimization results"""
    status: str
    objective_value: Optional[float] = None
    variables: Dict[str, float] = {}
    solve_time: Optional[float] = None
    total_time: Optional[float] = None
    solver: Optional[str] = None
    message: Optional[str] = None
    parameters_used: Dict[str, Any] = {}
    solver_info: SolverDiagnostics = SolverDiagnostics()
    
    class Config:
        schema_extra = {
            "example": {
                "status": "optimal",
                "objective_value": 1.0,
                "variables": {
                    "x1": 1.0,
                    "x2": 0.0
                },
                "solve_time": 0.0003,
                "total_time": 0.0015,
                "solver": "gurobi",
                "parameters_used": {
                    "max_time": 300,
                    "threads": 4
                },
                "solver_info": {
                    "iterations": 3,
                    "nodes": 1,
                    "gap": 0.0
                }
            }
        }


class JobSubmissionResponse(BaseModel):
    """Response when submitting a job"""
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    """Response when checking job status"""
    job_id: str
    status: JobStatus
    submitted_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[OptimizationResponse] = None
    error: Optional[str] = None


class MPSJobStatusResponse(BaseModel):
    """Response when checking MPS job status"""
    job_id: str
    status: JobStatus
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