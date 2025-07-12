#!/usr/bin/env python3
"""
ROLEX Solver Verification Script

This script performs comprehensive testing of all ROLEX solvers:
- Gurobi
- cuOpt
- SciPy

It provides detailed diagnostics and troubleshooting information.
"""

import sys
import traceback
import json
import time
import os
from typing import Dict, Any, List, Tuple

def log_info(message: str):
    """Log an info message"""
    print(f"[INFO] {message}")

def log_error(message: str):
    """Log an error message"""
    print(f"[ERROR] {message}")

def log_success(message: str):
    """Log a success message"""
    print(f"[SUCCESS] {message}")

def log_warning(message: str):
    """Log a warning message"""
    print(f"[WARNING] {message}")

def verify_environment():
    """Verify the Python environment"""
    log_info("Verifying Python environment...")
    
    # Check Python version
    log_info(f"Python version: {sys.version}")
    
    # Check if we're in a conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        log_info(f"Conda environment: {conda_env}")
    else:
        log_warning("Not in a conda environment")
    
    # Check library paths
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    if ld_library_path:
        log_info(f"LD_LIBRARY_PATH: {ld_library_path}")
    else:
        log_warning("LD_LIBRARY_PATH not set")
    
    log_success("Environment verification completed")

def verify_cuda_support():
    """Verify CUDA support through PyTorch"""
    log_info("Verifying CUDA support...")
    
    try:
        import torch
        log_info(f"PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            log_success("CUDA is available")
            log_info(f"CUDA version: {torch.version.cuda}")
            log_info(f"Number of GPUs: {torch.cuda.device_count()}")
            
            # Check GPU details
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_capability = torch.cuda.get_device_capability(i)
                log_info(f"GPU {i}: {gpu_name} (Capability: {gpu_capability})")
                
        else:
            log_error("CUDA is not available")
            return False
            
    except ImportError:
        log_error("PyTorch not installed")
        return False
    except Exception as e:
        log_error(f"Error verifying CUDA: {e}")
        return False
    
    log_success("CUDA verification completed")
    return True

def verify_gurobi() -> Tuple[bool, Dict[str, Any]]:
    """Verify Gurobi solver"""
    log_info("Verifying Gurobi solver...")
    
    result = {
        "available": False,
        "version": None,
        "license_valid": False,
        "error": None
    }
    
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        # Get version
        result["version"] = gp.gurobi.version()
        log_info(f"Gurobi version: {result['version']}")
        
        # Test license by creating a model
        model = gp.Model("test")
        x = model.addVar(name="x")
        y = model.addVar(name="y")
        model.setObjective(x + y, GRB.MAXIMIZE)
        model.addConstr(x + y <= 1, "constraint")
        
        # Solve
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            result["available"] = True
            result["license_valid"] = True
            log_success("Gurobi solver working correctly")
            log_info(f"Optimal solution: x={x.x:.4f}, y={y.x:.4f}, obj={model.objVal:.4f}")
        else:
            result["error"] = f"Optimization failed with status {model.status}"
            log_error(result["error"])
            
    except ImportError:
        result["error"] = "Gurobi not installed"
        log_error(result["error"])
    except Exception as e:
        result["error"] = str(e)
        log_error(f"Gurobi error: {e}")
        if "license" in str(e).lower():
            result["license_valid"] = False
    
    return result["available"], result

def verify_cuopt() -> Tuple[bool, Dict[str, Any]]:
    """Verify cuOpt solver"""
    log_info("Verifying cuOpt solver...")
    
    result = {
        "available": False,
        "version": None,
        "gpu_required": True,
        "error": None,
        "import_successful": False
    }
    
    try:
        # Test cuOpt import
        from cuopt import routing
        result["import_successful"] = True
        result["version"] = routing.__version__
        log_success("cuOpt import successful")
        log_info(f"cuOpt version: {result['version']}")
        
        # Test basic functionality
        log_info("Testing cuOpt basic functionality...")
        
        # Create a simple routing problem
        n_locations = 4
        n_vehicles = 1
        
        # Distance matrix (simple example)
        distance_matrix = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ]
        
        # Create routing model
        routing_model = routing.Model(n_locations, n_vehicles)
        routing_model.add_distance_matrix(distance_matrix)
        
        # Set start and end locations
        routing_model.set_start_location(0)
        routing_model.set_end_location(0)
        
        # Solve
        solution = routing_model.solve()
        
        if solution is not None:
            result["available"] = True
            log_success("cuOpt solver working correctly")
            log_info(f"Solution found with cost: {solution.get_cost()}")
        else:
            result["error"] = "cuOpt solve returned None"
            log_error(result["error"])
            
    except ImportError as e:
        result["error"] = f"cuOpt import failed: {e}"
        log_error(result["error"])
    except Exception as e:
        result["error"] = str(e)
        log_error(f"cuOpt error: {e}")
    
    return result["available"], result

def verify_scipy() -> Tuple[bool, Dict[str, Any]]:
    """Verify SciPy solver"""
    log_info("Verifying SciPy solver...")
    
    result = {
        "available": False,
        "version": None,
        "error": None
    }
    
    try:
        import scipy.optimize
        import numpy as np
        
        result["version"] = scipy.__version__
        log_info(f"SciPy version: {result['version']}")
        
        # Test linear programming
        c = np.array([1, 1])  # Objective coefficients
        A_ub = np.array([[1, 1]])  # Inequality constraint matrix
        b_ub = np.array([1])  # Inequality constraint bounds
        bounds = [(0, None), (0, None)]  # Variable bounds
        
        res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            result["available"] = True
            log_success("SciPy solver working correctly")
            log_info(f"Optimal solution: x={res.x}, fun={res.fun}")
        else:
            result["error"] = f"SciPy optimization failed: {res.message}"
            log_error(result["error"])
            
    except ImportError:
        result["error"] = "SciPy not installed"
        log_error(result["error"])
    except Exception as e:
        result["error"] = str(e)
        log_error(f"SciPy error: {e}")
    
    return result["available"], result

def verify_ommx():
    """Verify OMMX installation"""
    log_info("Verifying OMMX installation...")
    
    try:
        import ommx
        log_info(f"OMMX version: {ommx.__version__}")
        
        # Test basic OMMX functionality
        x1 = ommx.DecisionVariable.continuous(id=1, name="x1", lower=0, upper=10)
        x2 = ommx.DecisionVariable.continuous(id=2, name="x2", lower=0, upper=10)
        
        constraint = ommx.Constraint.linear(
            id=1,
            name="constraint1",
            terms=[ommx.LinearTerm(variable=1, coefficient=1.0),
                   ommx.LinearTerm(variable=2, coefficient=1.0)],
            equality=False,
            upper=1.0
        )
        
        objective = ommx.Objective.linear(
            id=1,
            name="objective",
            terms=[ommx.LinearTerm(variable=1, coefficient=1.0),
                   ommx.LinearTerm(variable=2, coefficient=1.0)],
            sense=ommx.OptimizationSense.MAXIMIZE
        )
        
        instance = ommx.Instance.from_components(
            decision_variables=[x1, x2],
            constraints=[constraint],
            objectives=[objective]
        )
        
        # Serialize and deserialize
        instance_bytes = instance.to_bytes()
        reconstructed = ommx.Instance.from_bytes(instance_bytes)
        
        log_success("OMMX working correctly")
        log_info(f"OMMX instance created with {len(instance_bytes)} bytes")
        
    except ImportError:
        log_error("OMMX not installed")
        return False
    except Exception as e:
        log_error(f"OMMX error: {e}")
        return False
    
    return True

def run_comprehensive_test():
    """Run comprehensive solver verification"""
    log_info("Starting comprehensive solver verification...")
    print("=" * 60)
    
    # Environment verification
    verify_environment()
    print("-" * 60)
    
    # CUDA verification
    cuda_available = verify_cuda_support()
    print("-" * 60)
    
    # OMMX verification
    ommx_available = verify_ommx()
    print("-" * 60)
    
    # Solver verification
    results = {}
    
    # Gurobi
    gurobi_available, gurobi_result = verify_gurobi()
    results["gurobi"] = gurobi_result
    print("-" * 60)
    
    # cuOpt
    cuopt_available, cuopt_result = verify_cuopt()
    results["cuopt"] = cuopt_result
    print("-" * 60)
    
    # SciPy
    scipy_available, scipy_result = verify_scipy()
    results["scipy"] = scipy_result
    print("-" * 60)
    
    # Summary
    log_info("SOLVER VERIFICATION SUMMARY")
    print("=" * 60)
    log_info(f"Environment: {'✅ OK' if True else '❌ FAIL'}")
    log_info(f"CUDA: {'✅ Available' if cuda_available else '❌ Not Available'}")
    log_info(f"OMMX: {'✅ Available' if ommx_available else '❌ Not Available'}")
    log_info(f"Gurobi: {'✅ Available' if gurobi_available else '❌ Not Available'}")
    log_info(f"cuOpt: {'✅ Available' if cuopt_available else '❌ Not Available'}")
    log_info(f"SciPy: {'✅ Available' if scipy_available else '❌ Not Available'}")
    
    # Save results to file
    with open("solver_verification_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    log_info("Results saved to solver_verification_results.json")
    
    return results

if __name__ == "__main__":
    try:
        results = run_comprehensive_test()
        
        # Exit with appropriate code
        total_solvers = 3
        available_solvers = sum(1 for solver in results.values() if solver.get("available", False))
        
        if available_solvers == total_solvers:
            log_success("All solvers verified successfully!")
            sys.exit(0)
        elif available_solvers > 0:
            log_warning(f"Only {available_solvers}/{total_solvers} solvers available")
            sys.exit(1)
        else:
            log_error("No solvers available")
            sys.exit(2)
            
    except Exception as e:
        log_error(f"Verification failed: {e}")
        traceback.print_exc()
        sys.exit(3) 