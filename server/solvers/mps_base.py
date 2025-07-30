"""
ROLEX Server - Base MPS Solver Interface
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import os
import tempfile
from models import MPSOptimizationResponse, SolverDiagnostics


class BaseMPSSolver(ABC):
    """Abstract base class for all MPS optimization solvers"""
    
    def __init__(self, name: str):
        self.name = name
        self._available = None
        self._solver_info = None
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if solver is available and ready to use"""
        pass
    
    @abstractmethod
    def get_solver_info(self) -> Dict[str, Any]:
        """Get detailed information about the solver"""
        pass
    
    @abstractmethod
    def solve_mps(self, mps_file_path: str, parameters: Dict[str, Any]) -> MPSOptimizationResponse:
        """
        Solve MPS optimization problem
        
        Args:
            mps_file_path: Path to the MPS file
            parameters: Solver-specific parameters
            
        Returns:
            MPSOptimizationResponse with solution details
            
        Raises:
            RuntimeError: If solver is not available or solving fails
        """
        pass
    
    def solve_with_timing(self, mps_file_path: str, parameters: Dict[str, Any]) -> MPSOptimizationResponse:
        """
        Wrapper that provides consistent response format and error handling
        
        Returns:
            MPSOptimizationResponse with timing information
        """
        print(f"DEBUG: {self.name} solve_with_timing called")
        
        if not self.is_available():
            return MPSOptimizationResponse(
                status="failed",
                objective_value=None,
                variables={},
                solve_time=0.0,
                total_time=0.0,
                solver=self.name,
                message=f"Solver {self.name} is not available",
                parameters_used=parameters,
                solver_info=SolverDiagnostics()
            )
        
        overall_start_time = time.time()
        
        try:
            # Validate MPS file exists
            if not os.path.exists(mps_file_path):
                raise RuntimeError(f"MPS file not found: {mps_file_path}")
            
            # Call solver-specific implementation
            result = self.solve_mps(mps_file_path, parameters)
            
            # Calculate total time
            total_time = time.time() - overall_start_time
            result.total_time = total_time
            result.parameters_used = parameters
            
            print(f"DEBUG: {self.name} solve completed - status: {result.status}")
            return result
            
        except Exception as e:
            total_time = time.time() - overall_start_time
            print(f"DEBUG: {self.name} solve failed - error: {str(e)}")
            
            return MPSOptimizationResponse(
                status="failed",
                objective_value=None,
                variables={},
                solve_time=0.0,
                total_time=total_time,
                solver=self.name,
                message=f"Solver error: {str(e)}",
                parameters_used=parameters,
                solver_info=SolverDiagnostics()
            )
    
    def create_temp_mps_file(self, mps_content: bytes) -> str:
        """
        Create a temporary MPS file from byte content
        
        Args:
            mps_content: MPS file content as bytes
            
        Returns:
            Path to temporary MPS file
        """
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mps', prefix='rolex_')
        
        try:
            with os.fdopen(temp_fd, 'wb') as f:
                f.write(mps_content)
            return temp_path
        except Exception as e:
            os.close(temp_fd)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise RuntimeError(f"Failed to create temporary MPS file: {str(e)}")
    
    def cleanup_temp_file(self, file_path: str):
        """
        Clean up temporary file
        
        Args:
            file_path: Path to file to remove
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"DEBUG: Cleaned up temporary file: {file_path}")
        except Exception as e:
            print(f"WARNING: Failed to cleanup temporary file {file_path}: {str(e)}")
    
    def _validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize solver parameters
        
        Args:
            parameters: Raw parameters from request
            
        Returns:
            Validated and normalized parameters
        """
        validated = {}
        
        # Common parameter validation
        if 'max_time' in parameters:
            max_time = parameters['max_time']
            if isinstance(max_time, (int, float)) and max_time > 0:
                validated['max_time'] = float(max_time)
        
        if 'threads' in parameters:
            threads = parameters['threads']
            if isinstance(threads, int) and threads > 0:
                validated['threads'] = threads
        
        if 'gap_tolerance' in parameters:
            gap = parameters['gap_tolerance']
            if isinstance(gap, (int, float)) and 0 <= gap <= 1:
                validated['gap_tolerance'] = float(gap)
        
        if 'verbose' in parameters:
            validated['verbose'] = bool(parameters['verbose'])
        
        if 'log_frequency' in parameters:
            log_freq = parameters['log_frequency']
            if isinstance(log_freq, (int, float)) and log_freq > 0:
                validated['log_frequency'] = float(log_freq)

        return validated
    
    def __str__(self) -> str:
        return f"{self.name} MPS Solver (available: {self.is_available()})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', available={self.is_available()})>" 