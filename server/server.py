"""
ROLEX Server - HTTP Server
"""

import logging
import asyncio
import tempfile
import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, status, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import uvicorn

from models import (
    OptimizationRequest, OptimizationResponse,
    JobSubmissionResponse, JobStatusResponse, ServerInfo,
    MPSOptimizationRequest, MPSJobStatusResponse,
    MPSSolverType,
    SERVER_NAME, SERVER_VERSION, SERVER_DESCRIPTION
)
from job_manager import JobManager

# Configure logging (output goes to stdout/stderr, captured by management script)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global solver manager instance
solver_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    global solver_manager
    
    logger.info(f"Starting {SERVER_NAME} {SERVER_VERSION}...")
    max_workers = getattr(app.state, 'max_workers', 1)
    solver_manager = JobManager(max_workers=max_workers)
    logger.info(f"{SERVER_NAME} {SERVER_VERSION} started successfully")
    yield
    
    # Shutdown
    logger.info(f"Shutting down {SERVER_NAME} {SERVER_VERSION}...")
    if solver_manager:
        solver_manager.shutdown()
    logger.info(f"{SERVER_NAME} {SERVER_VERSION} shutdown complete")

# FastAPI app
app = FastAPI(
    title=SERVER_NAME,
    description=SERVER_DESCRIPTION,
    version=SERVER_VERSION,
    lifespan=lifespan
)


@app.get("/", response_model=ServerInfo)
async def get_server_info():
    """Get server information and status"""
    if solver_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    stats = solver_manager.get_server_stats()
    
    return ServerInfo(
        available_solvers=stats["available_solvers"],
        active_jobs=stats["active_jobs"],
        total_jobs=stats["total_jobs"]
    )


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    if solver_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    return {"status": "healthy", "message": f"{SERVER_NAME} {SERVER_VERSION} is running"}


@app.get("/solvers")
async def get_solvers():
    """Get available solvers and their status"""
    if solver_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    return solver_manager.get_available_solvers()


@app.get("/solvers/mps")
async def get_mps_solvers():
    """Get available MPS solvers and their status"""
    if solver_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    return solver_manager.get_available_mps_solvers()


@app.post("/jobs/submit", response_model=JobSubmissionResponse)
async def submit_job(request: OptimizationRequest):
    """Submit an optimization job"""
    if solver_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    try:
        # Submit job to solver manager
        job_id = solver_manager.submit_job(
            solver_type=request.solver,
            model=request.model,
            parameters=request.parameters
        )
        
        return JobSubmissionResponse(
            job_id=job_id,
            status="queued",
            message=f"Job submitted successfully with {request.solver.value} solver"
        )
        
    except ValueError as e:
        # Model validation or solver availability errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Unexpected errors
        logger.error(f"Failed to submit job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit job"
        )


@app.post("/jobs/submit-mps", response_model=JobSubmissionResponse)
async def submit_mps_job(
    mps_file: UploadFile = File(...),
    solver: str = Form(...),
    parameters: str = Form(default="{}")
):
    """Submit an MPS optimization job"""
    if solver_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    # Validate file type
    if not mps_file.filename.endswith('.mps'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an MPS file (.mps extension)"
        )
    
    # Validate solver type
    try:
        solver_type = MPSSolverType(solver)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid solver type: {solver}. Available: {[s.value for s in MPSSolverType]}"
        )
    
    # Parse parameters
    try:
        params = json.loads(parameters)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON in parameters field"
        )
    
    # Create temporary file for MPS content
    try:
        # Read file content
        mps_content = await mps_file.read()
        
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mps', prefix='rolex_mps_')
        
        try:
            with os.fdopen(temp_fd, 'wb') as f:
                f.write(mps_content)
            
            # Submit job to solver manager
            job_id = solver_manager.submit_mps_job(
                solver_type=solver_type,
                mps_file_path=temp_path,
                parameters=params
            )
            
            return JobSubmissionResponse(
                job_id=job_id,
                status="queued",
                message=f"MPS job submitted successfully with {solver_type.value} solver"
            )
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
            
    except ValueError as e:
        # Model validation or solver availability errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Unexpected errors
        logger.error(f"Failed to submit MPS job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit MPS job"
        )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a specific job"""
    if solver_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    job_status = solver_manager.get_job_status(job_id)
    
    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    return job_status


@app.get("/jobs/{job_id}/mps", response_model=MPSJobStatusResponse)
async def get_mps_job_status(job_id: str):
    """Get the status of a specific MPS job"""
    if solver_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    job_status = solver_manager.get_mps_job_status(job_id)
    
    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MPS Job {job_id} not found"
        )
    
    return job_status


@app.get("/jobs")
async def list_jobs():
    """List all jobs (simplified endpoint)"""
    if solver_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    # Return basic statistics
    stats = solver_manager.get_server_stats()
    return {
        "total_jobs": stats["total_jobs"],
        "active_jobs": stats["active_jobs"],
        "message": "Use /jobs/{job_id} to get specific job details"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import argparse
    import uvicorn
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description=f"{SERVER_NAME} {SERVER_VERSION}")
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to run the server on (default: 8000)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=1, 
        help="Maximum number of concurrent solver instances (default: 1)"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    args = parser.parse_args()
    
    # Store max_workers in app state for lifespan function
    app.state.max_workers = args.max_workers
    
    logger.info(f"Starting {SERVER_NAME} {SERVER_VERSION} on {args.host}:{args.port}...")
    logger.info(f"Maximum concurrent solver instances: {args.max_workers}")
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    ) 