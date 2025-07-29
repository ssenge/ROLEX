"""
ROLEX Server - HTTP Server
MPS-only optimization server
"""

import logging
import asyncio
import tempfile
import os
import json
import traceback
import gzip
import subprocess
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, status, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import uvicorn

from models import (
    JobSubmissionResponse, ServerInfo,
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
    logger.info(f"{SERVER_NAME} {SERVER_VERSION} shutting down...")
    if solver_manager:
        await solver_manager.shutdown()
    logger.info(f"{SERVER_NAME} {SERVER_VERSION} shutdown complete")

# Create FastAPI app with lifespan
app = FastAPI(
    title=SERVER_NAME,
    description=SERVER_DESCRIPTION,
    version=SERVER_VERSION,
    lifespan=lifespan
)

# Root endpoint
@app.get("/", response_model=ServerInfo)
async def get_server_info():
    """Get server information and available solvers."""
    if solver_manager is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    mps_solvers = solver_manager.get_available_mps_solvers()
    
    return ServerInfo(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        status="running",
        available_solvers=mps_solvers,
        active_jobs=solver_manager.get_active_jobs_count(),
        total_jobs=solver_manager.get_total_jobs_count()
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "server": SERVER_NAME, "version": SERVER_VERSION}


# MPS Solvers endpoint
@app.get("/solvers/mps")
async def get_mps_solvers():
    """Get available MPS solvers."""
    if solver_manager is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    return solver_manager.get_available_mps_solvers()


# MPS Job submission endpoint
@app.post("/jobs/submit-mps", response_model=JobSubmissionResponse)
async def submit_mps_job(request: Request):
    """Submit an MPS optimization job."""
    if solver_manager is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    try:
        # Manually parse query parameters
        query_params = request.query_params
        solver = query_params.get("solver")
        filename = query_params.get("filename")
        parameters = query_params.get("parameters", "{}")

        if not solver or not filename:
            raise HTTPException(status_code=400, detail="'solver' and 'filename' are required query parameters.")

        # Parse solver type
        try:
            solver_type = MPSSolverType(solver)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid solver type: {solver}. Must be one of: {[s.value for s in MPSSolverType]}"
            )
        
        # Parse parameters
        try:
            params = json.loads(parameters)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in parameters")
        
        # Validate file type
        if not filename.lower().endswith(('.mps', '.mps.gz')):
            raise HTTPException(
                status_code=400,
                detail="File must be an MPS file (*.mps, *.mps.gz)"
            )
        
        # Create temp file and save uploaded MPS file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.mps', delete=False) as tmp_file:
            content = await request.body()
            
            # Check for gzip compression
            if request.headers.get("content-encoding") == "gzip" or filename.lower().endswith('.gz'):
                logger.info("Received gzipped file. Decompressing using gunzip CLI...")
                
                # Write gzipped content to a temporary .gz file
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.gz', delete=False) as tmp_gz_file:
                    tmp_gz_file.write(content)
                    tmp_gz_file_path = tmp_gz_file.name
                
                gzipped_size = os.path.getsize(tmp_gz_file_path)
                logger.info(f"Received gzipped file size: {gzipped_size / (1024*1024):.2f} MB")

                # Create a temporary file for the decompressed content
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.mps', delete=False) as tmp_decompressed_file:
                    decompressed_tmp_file_path = tmp_decompressed_file.name

                try:
                    # Execute gunzip to decompress the file
                    command = ["/usr/bin/gunzip", "-c", tmp_gz_file_path]
                    env = os.environ.copy()
                    if "/usr/bin" not in env.get("PATH", ""):
                        env["PATH"] = "/usr/bin:" + env.get("PATH", "")
                    process = subprocess.run(command, stdout=subprocess.PIPE, check=True, env=env)
                    content = process.stdout
                    
                    # Write the decompressed content to the temporary MPS file
                    with open(decompressed_tmp_file_path, 'wb') as f_decompressed:
                        f_decompressed.write(content)

                    decompressed_size = os.path.getsize(decompressed_tmp_file_path)
                    logger.info(f"Decompression complete. Decompressed to {decompressed_size / (1024*1024):.2f} MB.")
                    
                    # Update tmp_file_path to point to the decompressed file
                    tmp_file_path = decompressed_tmp_file_path

                except subprocess.CalledProcessError as e:
                    logger.error(f"Error during gunzip decompression: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
                    raise HTTPException(status_code=400, detail="Error during gunzip decompression.")
                finally:
                    # Clean up the temporary gzipped file
                    if os.path.exists(tmp_gz_file_path):
                        os.unlink(tmp_gz_file_path)
            else:
                # If not gzipped, write the content directly
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            # Ensure tmp_file_path is defined for the outer try-except block
            if 'tmp_file_path' not in locals():
                tmp_file_path = None # Initialize to None if not set above
            
            # This tmp_file.name is the one that will be passed to the solver manager
            # It should be the decompressed file if compression was detected, or the original if not.
            # The outer `with tempfile.NamedTemporaryFile` is no longer needed as we manage temp files manually.
            # So, we just need to ensure `tmp_file_path` is correctly set.
            # The original `tmp_file` context manager is removed, and `tmp_file_path` is set directly.
            # The content is already read into `content` variable.
            # The `tmp_file.write(content)` and `tmp_file_path = tmp_file.name` lines are moved/modified.
            # The `with tempfile.NamedTemporaryFile(mode='wb', suffix='.mps', delete=False) as tmp_file:`
            # block needs to be removed or refactored.
            # Let's refactor the whole block to make it cleaner.
            # The `tmp_file_path` should be set once, either to the decompressed file or the original content file.
            # The `content` variable will hold the final content to be written to the MPS file.
            
            # Refactored logic:
            final_mps_content = content # Assume content is already decompressed or original
            
            # Create temp file and save uploaded MPS file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.mps', delete=False) as tmp_file_obj:
                tmp_file_obj.write(final_mps_content)
                tmp_file_path = tmp_file_obj.name
        
        logger.info(f"Received MPS file: {filename} ({len(content)} bytes)")
        logger.info(f"Solver: {solver_type.value}, Parameters: {params}")
        
        # Create request object
        request = MPSOptimizationRequest(
            solver=solver_type,
            parameters=params
        )
        
        # Submit job
        job_id = await solver_manager.submit_mps_job(request, tmp_file_path)
        
        return JobSubmissionResponse(
            job_id=job_id,
            status="queued",
            message="Job submitted successfully"
        )
        
    except HTTPException:
        # Clean up temp file on error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except OSError:
                pass
        raise
    except Exception as e:
        # Clean up temp file on error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except OSError:
                pass
        
        # Force immediate console output
        print(f"=== MPS JOB SUBMISSION ERROR ===")
        print(f"Error: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Full traceback:")
        print(traceback.format_exc())
        print(f"=== END ERROR ===")
        
        logger.error(f"Error submitting MPS job: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting job: {str(e)}"
        )


# MPS Job status endpoint
@app.get("/jobs/{job_id}/mps", response_model=MPSJobStatusResponse)
async def get_mps_job_status(job_id: str):
    """Get the status of an MPS job."""
    if solver_manager is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    try:
        job_status = await solver_manager.get_mps_job_status(job_id)
        return job_status
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    except Exception as e:
        logger.error(f"Error getting MPS job status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting job status: {str(e)}"
        )


# List jobs endpoint
@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    if solver_manager is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    jobs = await solver_manager.list_jobs()
    return {"jobs": jobs}


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled exceptions."""
    error_msg = f"Unhandled exception: {str(exc)}"
    logger.error(error_msg)
    logger.error(f"Exception type: {type(exc).__name__}")
    logger.error(f"Request URL: {request.url}")
    logger.error(f"Request method: {request.method}")
    logger.error(f"Full traceback:\n{traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


# Server startup function
def run_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """Run the ROLEX server."""
    # Store configuration in app state
    app.state.max_workers = workers
    
    # Configure uvicorn
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
    
    server = uvicorn.Server(config)
    return server.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ROLEX Optimization Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    logger.info(f"Starting {SERVER_NAME} {SERVER_VERSION} on {args.host}:{args.port}")
    logger.info(f"Workers: {args.workers}")
    
    run_server(host=args.host, port=args.port, workers=args.workers) 