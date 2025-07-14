#!/usr/bin/env python3
"""
Debug test script for ROLEX MPS functionality
This bypasses HTTP to test the core MPS functionality directly
"""

import sys
import traceback
import tempfile
import os
import asyncio
from pathlib import Path

# Add the server directory to the path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test all imports to make sure they work"""
    print("=== Testing imports ===")
    try:
        print("Importing models...")
        from models import MPSOptimizationRequest, MPSSolverType
        print("✓ models imported successfully")
        
        print("Importing job_manager...")
        from job_manager import JobManager
        print("✓ job_manager imported successfully")
        
        print("Importing solvers...")
        from solvers.gurobi_mps_solver import GurobiMPSSolver
        from solvers.cuopt_mps_solver import CuOptMPSSolver
        print("✓ solvers imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_solver_initialization():
    """Test solver initialization"""
    print("\n=== Testing solver initialization ===")
    try:
        print("Creating Gurobi solver...")
        from solvers.gurobi_mps_solver import GurobiMPSSolver
        gurobi_solver = GurobiMPSSolver()
        print(f"✓ Gurobi solver created: {gurobi_solver}")
        print(f"  Available: {gurobi_solver.is_available()}")
        print(f"  Info: {gurobi_solver.get_solver_info()}")
        
        print("Creating cuOpt solver...")
        from solvers.cuopt_mps_solver import CuOptMPSSolver
        cuopt_solver = CuOptMPSSolver()
        print(f"✓ cuOpt solver created: {cuopt_solver}")
        print(f"  Available: {cuopt_solver.is_available()}")
        print(f"  Info: {cuopt_solver.get_solver_info()}")
        
        return True
    except Exception as e:
        print(f"✗ Solver initialization failed: {e}")
        traceback.print_exc()
        return False

def test_job_manager():
    """Test job manager initialization"""
    print("\n=== Testing job manager ===")
    try:
        print("Creating JobManager...")
        from job_manager import JobManager
        job_manager = JobManager(max_workers=1)
        print(f"✓ JobManager created: {job_manager}")
        
        print("Getting available solvers...")
        available = job_manager.get_available_mps_solvers()
        print(f"✓ Available MPS solvers: {available}")
        
        return job_manager
    except Exception as e:
        print(f"✗ JobManager failed: {e}")
        traceback.print_exc()
        return None

def create_test_mps_file():
    """Create a simple test MPS file"""
    print("\n=== Creating test MPS file ===")
    
    # Simple linear program: maximize x1 + x2 subject to x1 + x2 <= 1, x1,x2 >= 0
    mps_content = """NAME          TESTPROBLEM
ROWS
 N  OBJ
 L  ROW1
COLUMNS
    X1        OBJ                 1   ROW1                1
    X2        OBJ                 1   ROW1                1
RHS
    RHS1      ROW1                1
BOUNDS
 PL X1
 PL X2
ENDATA
"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mps', delete=False) as f:
        f.write(mps_content)
        temp_path = f.name
    
    print(f"✓ Created test MPS file: {temp_path}")
    return temp_path

async def test_direct_mps_job(job_manager, mps_file_path):
    """Test submitting an MPS job directly"""
    print("\n=== Testing direct MPS job submission ===")
    try:
        from models import MPSOptimizationRequest, MPSSolverType
        import time
        
        # Test with Gurobi first
        print("Testing with Gurobi solver...")
        request = MPSOptimizationRequest(
            solver=MPSSolverType.GUROBI,
            parameters={"max_time": 10}
        )
        
        print(f"Request created: {request}")
        print(f"MPS file path: {mps_file_path}")
        
        # This should trigger the same code path as the HTTP endpoint
        job_id = await job_manager.submit_mps_job(request, mps_file_path)
        print(f"✓ Job submitted with ID: {job_id}")
        
        # Wait a bit and check status
        time.sleep(2)
        
        status_response = await job_manager.get_mps_job_status(job_id)
        print(f"Job status: {status_response}")
        
        return True
        
    except Exception as e:
        print(f"✗ Direct MPS job test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    print("ROLEX MPS Debug Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("Imports failed, exiting...")
        return 1
    
    # Test solver initialization
    if not test_solver_initialization():
        print("Solver initialization failed, exiting...")
        return 1
    
    # Test job manager
    job_manager = test_job_manager()
    if not job_manager:
        print("Job manager failed, exiting...")
        return 1
    
    # Create test MPS file
    mps_file_path = create_test_mps_file()
    
    try:
        # Test direct MPS job
        if not await test_direct_mps_job(job_manager, mps_file_path):
            print("Direct MPS job test failed")
            return 1
        
        print("\n✓ All tests passed!")
        return 0
        
    finally:
        # Clean up
        if os.path.exists(mps_file_path):
            os.unlink(mps_file_path)
            print(f"Cleaned up test file: {mps_file_path}")

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 