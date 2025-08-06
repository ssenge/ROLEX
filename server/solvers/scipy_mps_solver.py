"""
SciPy-based MPS solver for Linear Programming problems
"""
import logging
import time
from typing import Dict, Any, List, Optional
import numpy as np

from .mps_base import BaseMPSSolver
from models import MPSOptimizationResponse, SolverDiagnostics, SolverCapability

logger = logging.getLogger(__name__)

# Check if SciPy is available
SCIPY_AVAILABLE = False
try:
    import scipy.optimize
    from scipy.sparse import csr_matrix
    SCIPY_AVAILABLE = True
    logger.info("SciPy solver available. SciPy version: %s", scipy.__version__)
except ImportError as e:
    logger.warning("SciPy not available: %s", e)

class SciPyMPSSolver(BaseMPSSolver):
    """SciPy-based MPS solver for Linear Programming"""
    
    def __init__(self):
        super().__init__("SciPy LP")
        self.solver_type = "scipy-lp"
    
    def is_available(self) -> bool:
        """Check if SciPy is available"""
        return SCIPY_AVAILABLE
    
    def get_capabilities(self) -> List[SolverCapability]:
        """Get solver capabilities - SciPy only supports LP"""
        return [SolverCapability.LP]
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get SciPy solver information"""
        info = {
            "name": "SciPy LP",
            "available": self.is_available(),
            "capabilities": [cap.value for cap in self.get_capabilities()]
        }
        if self.is_available():
            import scipy
            info["version"] = scipy.__version__
        else:
            info["version"] = None
        return info
    
    def _parse_mps_file(self, mps_file_path: str) -> Dict[str, Any]:
        """
        Simple MPS parser for SciPy
        Returns: dict with c, A_ub, b_ub, A_eq, b_eq, bounds
        """
        variables = {}  # var_name -> index
        var_names = []  # index -> var_name
        constraints = {}  # constraint_name -> {'type': 'L'/'G'/'E', 'coeffs': {var_idx: coeff}, 'rhs': value}
        objective = {}  # var_idx -> coeff
        bounds = {}  # var_idx -> (lower, upper)
        
        current_section = None
        
        with open(mps_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('*'):
                    continue
                
                # Section headers
                if line.startswith(('NAME', 'OBJSENSE')):
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
                
                # Parse sections
                parts = line.split()
                if not parts:
                    continue
                
                if current_section == 'ROWS':
                    row_type = parts[0]
                    row_name = parts[1]
                    if row_type in ['L', 'G', 'E']:
                        constraints[row_name] = {'type': row_type, 'coeffs': {}, 'rhs': 0.0}
                    elif row_type == 'N':
                        # Objective row
                        pass
                
                elif current_section == 'COLUMNS':
                    if 'MARKER' in line:
                        # Skip integer markers for now (SciPy doesn't support MIP)
                        continue
                    
                    var_name = parts[0]
                    if var_name not in variables:
                        variables[var_name] = len(var_names)
                        var_names.append(var_name)
                        bounds[len(var_names) - 1] = (0, None)  # Default bounds
                    
                    var_idx = variables[var_name]
                    
                    # Process coefficient pairs
                    for i in range(1, len(parts), 2):
                        if i + 1 < len(parts):
                            constraint_name = parts[i]
                            coeff = float(parts[i + 1])
                            
                            if constraint_name in constraints:
                                constraints[constraint_name]['coeffs'][var_idx] = coeff
                            else:
                                # Assume it's the objective
                                objective[var_idx] = coeff
                
                elif current_section == 'RHS':
                    for i in range(2, len(parts), 2):
                        if i + 1 < len(parts):
                            constraint_name = parts[i]
                            rhs_value = float(parts[i + 1])
                            if constraint_name in constraints:
                                constraints[constraint_name]['rhs'] = rhs_value
                
                elif current_section == 'BOUNDS':
                    bound_type = parts[0]
                    var_name = parts[2]
                    if var_name in variables:
                        var_idx = variables[var_name]
                        if bound_type == 'LO':  # Lower bound
                            lower, upper = bounds.get(var_idx, (0, None))
                            bounds[var_idx] = (float(parts[3]), upper)
                        elif bound_type == 'UP':  # Upper bound
                            lower, upper = bounds.get(var_idx, (0, None))
                            bounds[var_idx] = (lower, float(parts[3]))
                        elif bound_type == 'FX':  # Fixed
                            value = float(parts[3])
                            bounds[var_idx] = (value, value)
                        elif bound_type == 'FR':  # Free
                            bounds[var_idx] = (None, None)
        
        # Convert to SciPy format
        n_vars = len(var_names)
        
        # Objective coefficients (for minimization)
        c = np.zeros(n_vars)
        for var_idx, coeff in objective.items():
            c[var_idx] = coeff
        
        # Inequality constraints (A_ub * x <= b_ub)
        ineq_constraints = [con for con in constraints.values() if con['type'] in ['L', 'G']]
        if ineq_constraints:
            A_ub_rows = []
            b_ub = []
            for con in ineq_constraints:
                row = np.zeros(n_vars)
                for var_idx, coeff in con['coeffs'].items():
                    if con['type'] == 'L':  # <=
                        row[var_idx] = coeff
                    else:  # G, convert to <=
                        row[var_idx] = -coeff
                A_ub_rows.append(row)
                b_ub.append(con['rhs'] if con['type'] == 'L' else -con['rhs'])
            A_ub = np.array(A_ub_rows)
            b_ub = np.array(b_ub)
        else:
            A_ub = None
            b_ub = None
        
        # Equality constraints (A_eq * x = b_eq)
        eq_constraints = [con for con in constraints.values() if con['type'] == 'E']
        if eq_constraints:
            A_eq_rows = []
            b_eq = []
            for con in eq_constraints:
                row = np.zeros(n_vars)
                for var_idx, coeff in con['coeffs'].items():
                    row[var_idx] = coeff
                A_eq_rows.append(row)
                b_eq.append(con['rhs'])
            A_eq = np.array(A_eq_rows)
            b_eq = np.array(b_eq)
        else:
            A_eq = None
            b_eq = None
        
        # Variable bounds
        scipy_bounds = []
        for i in range(n_vars):
            lower, upper = bounds.get(i, (0, None))
            scipy_bounds.append((lower, upper))
        
        return {
            'c': c,
            'A_ub': A_ub,
            'b_ub': b_ub,
            'A_eq': A_eq,
            'b_eq': b_eq,
            'bounds': scipy_bounds,
            'var_names': var_names,
            'n_constraints': len(constraints)
        }
    
    def solve_mps(self, mps_file_path: str, parameters: Dict[str, Any] = None, optimality_tolerance: Optional[float] = None) -> MPSOptimizationResponse:
        """Solve MPS problem using SciPy"""
        if not self.is_available():
            return MPSOptimizationResponse(
                status="error",
                message="SciPy not available",
                objective_value=None,
                variables={},
                solve_time=0.0,
                solver=self.name,
                solver_info=SolverDiagnostics(),
                num_constraints=0,
                parameters_used=parameters or {}
            )
        
        logger.info(f"SciPy solver solving MPS file: {mps_file_path}")
        
        try:
            # Parse MPS file
            problem_data = self._parse_mps_file(mps_file_path)
            
            # Set up parameters
            method = 'highs' if parameters and parameters.get('method') else 'highs'  # Default to HiGHS
            options = {}
            if parameters:
                if 'max_time' in parameters:
                    options['time_limit'] = parameters['max_time']
                if 'verbose' in parameters and parameters['verbose']:
                    options['disp'] = True
            
            logger.info(f"SciPy: Solving with {len(problem_data['var_names'])} variables, {problem_data['n_constraints']} constraints")
            
            # Solve
            start_time = time.time()
            result = scipy.optimize.linprog(
                c=problem_data['c'],
                A_ub=problem_data['A_ub'],
                b_ub=problem_data['b_ub'],
                A_eq=problem_data['A_eq'],
                b_eq=problem_data['b_eq'],
                bounds=problem_data['bounds'],
                method=method,
                options=options
            )
            solve_time = time.time() - start_time
            
            # Process results
            if result.success:
                if result.x is not None:
                    solver_status = "OPTIMAL"
                    objective_value = float(result.fun) if result.fun is not None else 0.0
                    variable_values = {
                        problem_data['var_names'][i]: float(result.x[i])
                        for i in range(len(result.x))
                    }
                    message = f"SciPy solved successfully using {method}"
                else:
                    solver_status = "ERROR"
                    objective_value = None
                    variable_values = {}
                    message = f"SciPy: No solution found"
            else:
                if result.status == 2:  # Infeasible
                    solver_status = "INFEASIBLE"
                elif result.status == 3:  # Unbounded
                    solver_status = "UNBOUNDED"
                else:
                    solver_status = "ERROR"
                objective_value = None
                variable_values = {}
                message = f"SciPy failed: {result.message}"
            
            logger.info(f"SciPy: Solved in {solve_time:.4f}s with status {solver_status}")
            
            return MPSOptimizationResponse(
                status=solver_status.lower(),
                message=message,
                objective_value=objective_value,
                variables=variable_values,
                solve_time=solve_time,
                solver=self.name,
                solver_info=SolverDiagnostics(),
                num_constraints=problem_data['n_constraints'],
                parameters_used=parameters or {}
            )
            
        except Exception as e:
            logger.error(f"SciPy solver error: {e}")
            return MPSOptimizationResponse(
                status="error",
                message=f"SciPy solver error: {str(e)}",
                objective_value=None,
                variables={},
                solve_time=0.0,
                solver=self.name,
                solver_info=SolverDiagnostics(),
                num_constraints=0,
                parameters_used=parameters or {}
            ) 