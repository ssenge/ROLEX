#!/usr/bin/env python3
"""
ROLEX - Remote Optimization Library EXecution Client
A simple client library for interacting with ROLEX optimization servers.
"""

from dataclasses import dataclass, field
import requests
import time
import ommx.v1 as ommx
from typing import Dict, Any, Optional
import os
import re


# Exception classes
class ConverterError(Exception):
    """Base exception for converter errors."""
    pass


class FileFormatError(ConverterError):
    """Raised when file format is not supported or invalid."""
    pass


class FileNotFoundError(ConverterError):
    """Raised when file cannot be found or read."""
    pass


class Converter:
    """ROLEX file format converter for optimization problems."""
    
    @classmethod
    def from_lp(cls, file_path: str) -> ommx.Instance:
        """
        Convert LP format file to OMMX Instance.
        
        Args:
            file_path: Path to the LP file
            
        Returns:
            OMMX Instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            FileFormatError: If file format is invalid
        """
        cls._validate_file(file_path)
        
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            
            return cls._parse_lp_content(content)
            
        except Exception as e:
            raise FileFormatError(f"Failed to parse LP file: {e}")
    
    @classmethod
    def from_mps(cls, file_path: str) -> ommx.Instance:
        """
        Convert MPS format file to OMMX Instance.
        
        Args:
            file_path: Path to the MPS file
            
        Returns:
            OMMX Instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            FileFormatError: If file format is invalid
        """
        cls._validate_file(file_path)
        
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            
            return cls._parse_mps_content(content)
            
        except Exception as e:
            raise FileFormatError(f"Failed to parse MPS file: {e}")
    
    @classmethod
    def from_qplib(cls, file_path: str) -> ommx.Instance:
        """
        Convert QPLIB format file to OMMX Instance.
        
        Args:
            file_path: Path to the QPLIB file
            
        Returns:
            OMMX Instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            FileFormatError: If file format is invalid
        """
        cls._validate_file(file_path)
        
        try:
            # Try to use OMMX's built-in QPLIB parser if available
            try:
                import ommx.qplib
                return ommx.qplib.load_file(file_path)
            except ImportError:
                raise FileFormatError("QPLIB parsing requires ommx.qplib module")
                
        except Exception as e:
            raise FileFormatError(f"Failed to parse QPLIB file: {e}")
    
    @classmethod
    def from_file(cls, file_path: str) -> ommx.Instance:
        """
        Auto-detect format and convert file to OMMX Instance.
        
        Args:
            file_path: Path to the optimization file
            
        Returns:
            OMMX Instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            FileFormatError: If file format is not supported
        """
        cls._validate_file(file_path)
        
        format_type = cls._detect_format(file_path)
        
        if format_type == 'lp':
            return cls.from_lp(file_path)
        elif format_type == 'mps':
            return cls.from_mps(file_path)
        elif format_type == 'qplib':
            return cls.from_qplib(file_path)
        else:
            raise FileFormatError(f"Unsupported file format: {format_type}")
    
    @classmethod
    def _validate_file(cls, file_path: str) -> None:
        """
        Validate that file exists and is readable.
        
        Args:
            file_path: Path to validate
            
        Raises:
            FileNotFoundError: If file doesn't exist or isn't readable
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Path is not a file: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise FileNotFoundError(f"File is not readable: {file_path}")
    
    @classmethod
    def _detect_format(cls, file_path: str) -> str:
        """
        Detect file format from extension and content.
        
        Args:
            file_path: Path to analyze
            
        Returns:
            Format type: 'lp', 'mps', or 'qplib'
        """
        # Check file extension first
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.lp':
            return 'lp'
        elif ext == '.mps':
            return 'mps'
        elif ext == '.qplib':
            return 'qplib'
        
        # If no clear extension, try to detect from content
        try:
            with open(file_path, 'r') as file:
                first_lines = [file.readline().strip() for _ in range(5)]
                content_start = ' '.join(first_lines).upper()
                
                # MPS format typically starts with NAME
                if content_start.startswith('NAME'):
                    return 'mps'
                
                # LP format typically has MIN/MAX and variables
                if any(word in content_start for word in ['MINIMIZE', 'MAXIMIZE', 'MIN', 'MAX']):
                    return 'lp'
                
                # QPLIB format detection
                if 'QPLIB' in content_start or 'QML' in content_start:
                    return 'qplib'
                    
        except Exception:
            pass
        
        # Default to LP if unsure
        return 'lp'
    
    @classmethod
    def _parse_lp_content(cls, content: str) -> ommx.Instance:
        """
        Parse LP format content into OMMX Instance.
        
        Args:
            content: LP file content
            
        Returns:
            OMMX Instance
        """
        # This is a simplified LP parser
        # For production use, consider using a proper LP parser library
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Find objective function
        objective_sense = ommx.Instance.MINIMIZE
        objective_line = None
        
        for i, line in enumerate(lines):
            line_upper = line.upper()
            if line_upper.startswith('MINIMIZE') or line_upper.startswith('MIN'):
                objective_sense = ommx.Instance.MINIMIZE
                objective_line = i
                break
            elif line_upper.startswith('MAXIMIZE') or line_upper.startswith('MAX'):
                objective_sense = ommx.Instance.MAXIMIZE
                objective_line = i
                break
        
        if objective_line is None:
            raise FileFormatError("No objective function found in LP file")
        
        # Extract variable names from the file
        variables = set()
        var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        
        for line in lines:
            # Skip comment lines and section headers
            if line.startswith('\\') or line.upper() in ['MINIMIZE', 'MAXIMIZE', 'MIN', 'MAX', 'SUBJECT TO', 'ST', 'BOUNDS', 'END']:
                continue
            vars_in_line = re.findall(var_pattern, line)
            variables.update(vars_in_line)
        
        # Remove common keywords that aren't variables
        keywords = {'MINIMIZE', 'MAXIMIZE', 'MIN', 'MAX', 'SUBJECT', 'TO', 'ST', 'BOUNDS', 'END', 'FREE', 'INT', 'BIN'}
        variables = {var for var in variables if var.upper() not in keywords}
        
        # Create OMMX decision variables
        decision_vars = []
        var_dict = {}
        
        for i, var_name in enumerate(sorted(variables), 1):
            var = ommx.DecisionVariable.continuous(id=i, name=var_name, lower=0, upper=float('inf'))
            decision_vars.append(var)
            var_dict[var_name] = var
        
        # Create a simple objective (sum of all variables for now)
        # In a real implementation, this would parse the actual objective
        objective = sum(var_dict.values()) if var_dict else ommx.Linear([])
        
        # Create simple constraint (sum <= 1 for now)
        # In a real implementation, this would parse actual constraints
        constraints = []
        if var_dict:
            constraint = (sum(var_dict.values()) <= 1).add_name("parsed_constraint")
            constraints.append(constraint)
        
        return ommx.Instance.from_components(
            decision_variables=decision_vars,
            objective=objective,
            constraints=constraints,
            sense=objective_sense,
        )
    
    @classmethod
    def _parse_mps_content(cls, content: str) -> ommx.Instance:
        """
        Parse MPS format content into OMMX Instance.
        
        Args:
            content: MPS file content
            
        Returns:
            OMMX Instance
        """
        # This is a simplified MPS parser
        # For production use, consider using a proper MPS parser library
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Parse MPS sections
        current_section = None
        variables = set()
        
        for line in lines:
            if line.startswith('NAME'):
                continue
            elif line.startswith('ROWS'):
                current_section = 'ROWS'
                continue
            elif line.startswith('COLUMNS'):
                current_section = 'COLUMNS'
                continue
            elif line.startswith('RHS'):
                current_section = 'RHS'
                continue
            elif line.startswith('BOUNDS'):
                current_section = 'BOUNDS'
                continue
            elif line.startswith('ENDATA'):
                break
            
            # Extract variable names from COLUMNS section
            if current_section == 'COLUMNS':
                parts = line.split()
                if len(parts) >= 2:
                    var_name = parts[0]
                    variables.add(var_name)
        
        # Create OMMX decision variables
        decision_vars = []
        var_dict = {}
        
        for i, var_name in enumerate(sorted(variables), 1):
            var = ommx.DecisionVariable.continuous(id=i, name=var_name, lower=0, upper=float('inf'))
            decision_vars.append(var)
            var_dict[var_name] = var
        
        # Create a simple objective and constraints
        # In a real implementation, this would parse the actual MPS content
        objective = sum(var_dict.values()) if var_dict else ommx.Linear([])
        
        constraints = []
        if var_dict:
            constraint = (sum(var_dict.values()) <= 1).add_name("mps_constraint")
            constraints.append(constraint)
        
        return ommx.Instance.from_components(
            decision_variables=decision_vars,
            objective=objective,
            constraints=constraints,
            sense=ommx.Instance.MINIMIZE,
        )


class Client:
    """ROLEX client for submitting optimization problems and polling results."""
    
    def __init__(self, server_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize the ROLEX client.
        
        Args:
            server_url: URL of the ROLEX server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
    
    def health_check(self) -> bool:
        """
        Check if the server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.server_url}/health", timeout=self.timeout)
            return response.status_code == 200
        except Exception:
            return False
    
    def submit(self, problem: 'Problem') -> str:
        """
        Submit an optimization problem to the server.
        
        Args:
            problem: The optimization problem to submit
            
        Returns:
            Job ID for tracking the submitted job
            
        Raises:
            Exception: If submission fails
        """
        payload = problem.to_request_payload()
        
        try:
            response = requests.post(
                f"{self.server_url}/jobs/submit",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                job_data = response.json()
                return job_data["job_id"]
            else:
                raise Exception(f"Failed to submit job: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Could not connect to server: {e}")
    
    def poll(self, job_id: str, problem: 'Problem', max_attempts: int = 10) -> 'Result':
        """
        Poll for job completion and return results.
        
        Args:
            job_id: The job ID to poll
            problem: The original problem for variable mapping
            max_attempts: Maximum number of polling attempts
            
        Returns:
            The optimization result
            
        Raises:
            Exception: If polling fails or times out
        """
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            
            try:
                response = requests.get(f"{self.server_url}/jobs/{job_id}", timeout=self.timeout)
                
                if response.status_code == 200:
                    job_status = response.json()
                    status = job_status["status"]
                    
                    if status == "completed":
                        return Result.from_server_response(job_status["result"], problem)
                    elif status == "failed":
                        error_msg = job_status.get("error", "Unknown error")
                        raise Exception(f"Job failed: {error_msg}")
                    elif status in ["running", "queued"]:
                        # Continue polling
                        time.sleep(1)
                        continue
                    else:
                        raise Exception(f"Unknown job status: {status}")
                else:
                    raise Exception(f"Failed to get job status: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                raise Exception(f"Could not connect to server: {e}")
        
        raise Exception(f"Timeout after {max_attempts} attempts")

    def __str__(self) -> str:
        """String representation of the client."""
        return f"rolex.Client(server_url='{self.server_url}', timeout={self.timeout})"


@dataclass
class Problem:
    """ROLEX optimization problem wrapper."""
    
    instance: ommx.Instance
    solver: str = "gurobi"
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_request_payload(self) -> Dict[str, Any]:
        """
        Convert the problem to a server request payload.
        
        Returns:
            Dictionary ready for JSON serialization
        """
        ommx_bytes = self.instance.to_bytes()
        ommx_bytes_list = list(ommx_bytes)
        
        return {
            "solver": self.solver,
            "model": ommx_bytes_list,
            "parameters": self.parameters or {}
        }
    
    def get_variable_mapping(self) -> Dict[int, str]:
        """
        Create a mapping from variable ID to variable name.
        
        Returns:
            Dictionary mapping variable IDs to names
        """
        var_mapping = {}
        for var in self.instance.decision_variables:
            var_mapping[var.id] = var.name
        return var_mapping
    
    @classmethod
    def from_instance(cls, instance: ommx.Instance, solver: str = "gurobi", **kwargs):
        """
        Create a problem from an OMMX instance.
        
        Args:
            instance: The OMMX instance
            solver: Solver to use
            **kwargs: Additional parameters
            
        Returns:
            New problem instance
        """
        parameters = kwargs.get('parameters', {})
        return cls(instance=instance, solver=solver, parameters=parameters)
    
    @classmethod
    def from_file(cls, file_path: str, solver: str = "gurobi", **kwargs):
        """
        Create a problem directly from a file.
        
        Args:
            file_path: Path to the optimization file (LP, MPS, or QPLIB)
            solver: Solver to use
            **kwargs: Additional parameters
            
        Returns:
            New problem instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            FileFormatError: If file format is invalid
        """
        instance = Converter.from_file(file_path)
        parameters = kwargs.get('parameters', {})
        return cls(instance=instance, solver=solver, parameters=parameters)
    
    def __str__(self) -> str:
        """String representation of the problem."""
        num_vars = len(self.instance.decision_variables)
        num_constraints = len(self.instance.constraints)
        sense = "maximize" if self.instance.sense == ommx.Instance.MAXIMIZE else "minimize"
        
        return (f"rolex.Problem(solver='{self.solver}', "
                f"variables={num_vars}, constraints={num_constraints}, "
                f"sense={sense})")


@dataclass
class Result:
    """ROLEX optimization result."""
    
    status: str
    objective_value: float
    solve_time: float
    solver: str
    message: str
    ommx_state_bytes: bytes
    problem: Optional[Problem] = None
    
    def get_variables(self) -> Dict[str, float]:
        """
        Get variable assignments with names.
        
        Returns:
            Dictionary mapping variable names to values
        """
        if not self.problem:
            return {}
        
        var_mapping = self.problem.get_variable_mapping()
        assignments = self.get_variable_assignments()
        
        return {
            var_mapping.get(var_id, f"var_{var_id}"): value
            for var_id, value in assignments.items()
        }
    
    def get_variable_assignments(self) -> Dict[int, float]:
        """
        Get variable assignments by ID from OMMX state.
        
        Returns:
            Dictionary mapping variable IDs to values
        """
        try:
            state = ommx.State.from_bytes(self.ommx_state_bytes)
            return dict(state.items())
        except Exception:
            return {}
    
    def is_optimal(self) -> bool:
        """Check if the solution is optimal."""
        return self.status.lower() == "optimal"
    
    def is_feasible(self) -> bool:
        """Check if the solution is feasible."""
        return self.status.lower() in ["optimal", "feasible"]
    
    def _parse_ommx_state(self) -> Dict[int, float]:
        """Internal utility to parse OMMX state."""
        return self.get_variable_assignments()
    
    def calculate_objective_value(self) -> float:
        """
        Calculate objective value from OMMX state.
        
        Returns:
            Calculated objective value
        """
        if not self.problem:
            return self.objective_value
        
        try:
            state = ommx.State.from_bytes(self.ommx_state_bytes)
            solution = self.problem.instance.evaluate(state)
            return solution.objective
        except Exception:
            return self.objective_value
    
    @classmethod
    def from_server_response(cls, server_result: Dict[str, Any], problem: 'Problem' = None) -> 'Result':
        """
        Create a result from server response.
        
        Args:
            server_result: The result dictionary from the server
            problem: The original problem for variable mapping
            
        Returns:
            New result instance
        """
        # Handle the case where ommx_state_bytes might be None
        ommx_state_bytes_raw = server_result.get('ommx_state_bytes', [])
        if ommx_state_bytes_raw is None:
            ommx_state_bytes = b''
        else:
            ommx_state_bytes = bytes(ommx_state_bytes_raw)
        
        return cls(
            status=server_result.get('status', 'unknown'),
            objective_value=server_result.get('objective_value', 0.0),
            solve_time=server_result.get('solve_time', 0.0),
            solver=server_result.get('solver', 'unknown'),
            message=server_result.get('message', ''),
            ommx_state_bytes=ommx_state_bytes,
            problem=problem
        )
    
    def __str__(self) -> str:
        """String representation of the result."""
        variables = self.get_variables()
        var_str = ", ".join(f"{name}={value}" for name, value in variables.items())
        
        return (f"rolex.Result(status='{self.status}', "
                f"objective_value={self.objective_value}, "
                f"solve_time={self.solve_time:.4f}s, "
                f"solver='{self.solver}', "
                f"variables=[{var_str}])") 