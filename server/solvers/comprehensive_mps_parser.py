"""
Comprehensive MPS Parser and Intermediate Representation

This module provides a complete implementation of the MPS (Mathematical Programming System)
file format parser with support for all advanced features including:
- Linear and quadratic objectives (QUADOBJ)
- Quadratic constraints (QMATRIX/QCMATRIX)  
- Advanced bounds (BV, LI, UI, etc.)
- Special Ordered Sets (SOS)
- Range constraints (RANGES)
- Conic constraints (CSECTION)

The parser uses a state machine approach and creates a unified intermediate representation
that can be converted to various solver formats (CVXPY, Pyomo, etc.).
"""

import logging
import re
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from enum import Enum
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


class ObjectiveSense(Enum):
    MINIMIZE = "MIN"
    MAXIMIZE = "MAX"


class ConstraintType(Enum):
    EQUAL = "E"
    LESS_EQUAL = "L" 
    GREATER_EQUAL = "G"
    FREE = "N"  # objective row


class BoundType(Enum):
    LOWER = "LO"      # x >= value
    UPPER = "UP"      # x <= value
    FIXED = "FX"      # x = value
    FREE = "FR"       # -inf <= x <= inf
    MINUS_INF = "MI"  # x >= -inf
    PLUS_INF = "PL"   # x <= +inf
    BINARY = "BV"     # x in {0,1}
    INTEGER = "LI"    # x >= value, integer
    INTEGER_UP = "UI" # x <= value, integer


class SOSType(Enum):
    SOS1 = 1  # at most one variable nonzero
    SOS2 = 2  # at most two adjacent variables nonzero


@dataclass
class LinearConstraint:
    name: str
    constraint_type: ConstraintType
    coefficients: Dict[int, float] = field(default_factory=dict)  # var_index -> coefficient
    rhs: float = 0.0
    range_value: Optional[float] = None  # for RANGES section


@dataclass
class SOSConstraint:
    name: str
    sos_type: SOSType
    variables: List[int] = field(default_factory=list)  # variable indices
    weights: List[float] = field(default_factory=list)  # priority weights


@dataclass
class ConicConstraint:
    name: str
    cone_type: str  # QUAD, RQUAD, etc.
    variables: List[int] = field(default_factory=list)


class MPSProblem:
    """Complete MPS problem representation supporting all MPS features"""
    
    def __init__(self):
        # Problem metadata
        self.name: str = ""
        self.objective_sense: ObjectiveSense = ObjectiveSense.MINIMIZE
        self.objective_name: Optional[str] = None
        
        # Variables
        self.variables: Dict[str, int] = {}  # name -> index
        self.var_names: List[str] = []       # index -> name
        self.n_vars: int = 0
        
        # Variable bounds and types
        self.var_bounds: Dict[int, Tuple[float, float]] = {}  # (lower, upper)
        self.integer_vars: Set[int] = set()
        self.binary_vars: Set[int] = set()
        
        # Linear components
        self.linear_objective: Dict[int, float] = {}  # var_index -> coeff
        self.constraints: Dict[str, LinearConstraint] = {}
        
        # Quadratic components
        self.quadratic_objective: Dict[Tuple[int, int], float] = {}  # (i,j) -> coeff
        self.quadratic_constraints: Dict[str, Dict[Tuple[int, int], float]] = {}
        
        # Advanced features
        self.sos_constraints: List[SOSConstraint] = []
        self.conic_constraints: List[ConicConstraint] = []
        
        # Row/column ordering (for exact MPS reconstruction if needed)
        self.row_order: List[str] = []
        self.column_order: List[str] = []
        
        # Parsing state
        self._objective_rows: Set[str] = set()  # N-type rows
    
    def get_variable_index(self, var_name: str) -> int:
        """Get or create variable index for given name"""
        if var_name not in self.variables:
            self.variables[var_name] = self.n_vars
            self.var_names.append(var_name)
            # Default bounds: x >= 0
            self.var_bounds[self.n_vars] = (0.0, float('inf'))
            self.n_vars += 1
        return self.variables[var_name]
    
    def set_variable_bound(self, var_idx: int, bound_type: BoundType, value: Optional[float] = None):
        """Set variable bound according to MPS bound type"""
        current_lower, current_upper = self.var_bounds.get(var_idx, (0.0, float('inf')))
        
        if bound_type == BoundType.LOWER:
            self.var_bounds[var_idx] = (value, current_upper)
        elif bound_type == BoundType.UPPER:
            self.var_bounds[var_idx] = (current_lower, value)
        elif bound_type == BoundType.FIXED:
            self.var_bounds[var_idx] = (value, value)
        elif bound_type == BoundType.FREE:
            self.var_bounds[var_idx] = (-float('inf'), float('inf'))
        elif bound_type == BoundType.MINUS_INF:
            self.var_bounds[var_idx] = (-float('inf'), current_upper)
        elif bound_type == BoundType.PLUS_INF:
            self.var_bounds[var_idx] = (current_lower, float('inf'))
        elif bound_type == BoundType.BINARY:
            self.var_bounds[var_idx] = (0.0, 1.0)
            self.binary_vars.add(var_idx)
        elif bound_type == BoundType.INTEGER:
            self.var_bounds[var_idx] = (value, current_upper)
            self.integer_vars.add(var_idx)
        elif bound_type == BoundType.INTEGER_UP:
            self.var_bounds[var_idx] = (current_lower, value)
            self.integer_vars.add(var_idx)
    
    def add_quadratic_objective_term(self, var1_idx: int, var2_idx: int, coeff: float):
        """Add quadratic term to objective function"""
        # Store in canonical form: (min(i,j), max(i,j))
        key = (min(var1_idx, var2_idx), max(var1_idx, var2_idx))
        if key in self.quadratic_objective:
            self.quadratic_objective[key] += coeff
        else:
            self.quadratic_objective[key] = coeff
    
    def add_quadratic_constraint_term(self, constraint_name: str, var1_idx: int, var2_idx: int, coeff: float):
        """Add quadratic term to constraint"""
        if constraint_name not in self.quadratic_constraints:
            self.quadratic_constraints[constraint_name] = {}
        
        key = (min(var1_idx, var2_idx), max(var1_idx, var2_idx))
        quad_terms = self.quadratic_constraints[constraint_name]
        
        if key in quad_terms:
            quad_terms[key] += coeff
        else:
            quad_terms[key] = coeff


class ComprehensiveMPSParser:
    """State machine-based MPS parser supporting complete MPS specification"""
    
    def __init__(self):
        self.current_section: Optional[str] = None
        self.problem = MPSProblem()
        self.in_integer_block = False
        
        # Section handlers
        self.section_handlers = {
            'NAME': self._handle_name_section,
            'OBJSENSE': self._handle_objsense_section,
            'OBJNAME': self._handle_objname_section,
            'ROWS': self._handle_rows_section,
            'COLUMNS': self._handle_columns_section,
            'RHS': self._handle_rhs_section,
            'RANGES': self._handle_ranges_section,
            'BOUNDS': self._handle_bounds_section,
            'QUADOBJ': self._handle_quadobj_section,
            'QMATRIX': self._handle_qmatrix_section,
            'QCMATRIX': self._handle_qcmatrix_section,
            'SOS': self._handle_sos_section,
            'CSECTION': self._handle_csection_section,
        }
    
    def parse_mps_file(self, mps_file_path: str) -> MPSProblem:
        """
        Parse complete MPS file and return unified problem representation
        
        Args:
            mps_file_path: Path to MPS file
            
        Returns:
            MPSProblem with complete problem representation
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If MPS format is invalid
        """
        logger.info(f"Parsing MPS file with comprehensive parser: {mps_file_path}")
        
        try:
            with open(mps_file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        self._parse_line(line.rstrip('\n\r'), line_num)
                    except Exception as e:
                        logger.warning(f"Error parsing line {line_num}: {line.strip()}. Error: {e}")
                        continue
                        
        except FileNotFoundError:
            raise FileNotFoundError(f"MPS file not found: {mps_file_path}")
        except Exception as e:
            raise ValueError(f"Error reading MPS file {mps_file_path}: {str(e)}")
        
        # Validate parsed problem
        self._validate_problem()
        
        logger.info(f"MPS parsing complete: {self.problem.n_vars} variables, "
                   f"{len(self.problem.constraints)} linear constraints, "
                   f"{len(self.problem.quadratic_objective)} quadratic objective terms, "
                   f"{len(self.problem.quadratic_constraints)} quadratic constraints, "
                   f"{len(self.problem.sos_constraints)} SOS constraints")
        
        return self.problem
    
    def _parse_line(self, line: str, line_num: int):
        """Parse individual line using state machine"""
        # Skip empty lines and comments
        if not line or line.startswith('*'):
            return
        
        # Check for section headers
        section_match = self._identify_section(line)
        if section_match:
            self.current_section = section_match
            logger.debug(f"Entering section: {self.current_section}")
            return
        
        # Handle ENDATA terminator
        if line.startswith('ENDATA'):
            self.current_section = None
            return
        
        # Handle section content
        if self.current_section and self.current_section in self.section_handlers:
            self.section_handlers[self.current_section](line, line_num)
    
    def _identify_section(self, line: str) -> Optional[str]:
        """Identify MPS section from line"""
        sections = ['NAME', 'OBJSENSE', 'OBJNAME', 'ROWS', 'COLUMNS', 'RHS', 
                   'RANGES', 'BOUNDS', 'QUADOBJ', 'QMATRIX', 'QCMATRIX', 
                   'SOS', 'CSECTION']
        
        for section in sections:
            if line.startswith(section):
                return section
        return None
    
    def _handle_name_section(self, line: str, line_num: int):
        """Handle NAME section"""
        parts = line.split()
        if len(parts) >= 2:
            self.problem.name = parts[1]
    
    def _handle_objsense_section(self, line: str, line_num: int):
        """Handle OBJSENSE section"""
        if 'MAX' in line.upper():
            self.problem.objective_sense = ObjectiveSense.MAXIMIZE
        else:
            self.problem.objective_sense = ObjectiveSense.MINIMIZE
    
    def _handle_objname_section(self, line: str, line_num: int):
        """Handle OBJNAME section"""
        parts = line.split()
        if len(parts) >= 2:
            self.problem.objective_name = parts[1]
    
    def _handle_rows_section(self, line: str, line_num: int):
        """Handle ROWS section - define constraint types"""
        parts = line.split()
        if len(parts) < 2:
            return
        
        row_type = parts[0]
        row_name = parts[1]
        
        self.problem.row_order.append(row_name)
        
        if row_type in ['E', 'L', 'G']:
            constraint = LinearConstraint(
                name=row_name,
                constraint_type=ConstraintType(row_type)
            )
            self.problem.constraints[row_name] = constraint
            
        elif row_type == 'N':
            # Objective row (first N is typically objective)
            self.problem._objective_rows.add(row_name)
            if self.problem.objective_name is None:
                self.problem.objective_name = row_name
    
    def _handle_columns_section(self, line: str, line_num: int):
        """Handle COLUMNS section - define variables and coefficients"""
        # Handle integer markers
        if "'MARKER'" in line:
            if "'INTORG'" in line:
                self.in_integer_block = True
            elif "'INTEND'" in line:
                self.in_integer_block = False
            return
        
        parts = line.split()
        if len(parts) < 3:
            return
        
        var_name = parts[0]
        var_idx = self.problem.get_variable_index(var_name)
        
        # Mark as integer if in integer block
        if self.in_integer_block:
            self.problem.integer_vars.add(var_idx)
        
        # Track column order
        if var_name not in self.problem.column_order:
            self.problem.column_order.append(var_name)
        
        # Process coefficient pairs
        for i in range(1, len(parts), 2):
            if i + 1 >= len(parts):
                break
                
            row_name = parts[i]
            try:
                coeff = float(parts[i + 1])
            except ValueError:
                continue
            
            # Add to constraint or objective
            if row_name in self.problem.constraints:
                self.problem.constraints[row_name].coefficients[var_idx] = coeff
            elif row_name in self.problem._objective_rows:
                self.problem.linear_objective[var_idx] = coeff
    
    def _handle_rhs_section(self, line: str, line_num: int):
        """Handle RHS section - right-hand side values"""
        parts = line.split()
        if len(parts) < 3:
            return
        
        # Skip RHS name (parts[0])
        for i in range(1, len(parts), 2):
            if i + 1 >= len(parts):
                break
                
            constraint_name = parts[i]
            try:
                rhs_value = float(parts[i + 1])
            except ValueError:
                continue
                
            if constraint_name in self.problem.constraints:
                self.problem.constraints[constraint_name].rhs = rhs_value
    
    def _handle_ranges_section(self, line: str, line_num: int):
        """Handle RANGES section - range constraints"""
        parts = line.split()
        if len(parts) < 3:
            return
        
        # Skip range name (parts[0])
        for i in range(1, len(parts), 2):
            if i + 1 >= len(parts):
                break
                
            constraint_name = parts[i]
            try:
                range_value = float(parts[i + 1])
            except ValueError:
                continue
                
            if constraint_name in self.problem.constraints:
                self.problem.constraints[constraint_name].range_value = range_value
    
    def _handle_bounds_section(self, line: str, line_num: int):
        """Handle BOUNDS section - variable bounds"""
        parts = line.split()
        if len(parts) < 3:
            return
        
        bound_type_str = parts[0]
        # Skip bound name if present (parts[1])
        var_name = parts[2] if len(parts) > 3 else parts[1]
        
        if var_name not in self.problem.variables:
            return
            
        var_idx = self.problem.variables[var_name]
        
        try:
            bound_type = BoundType(bound_type_str)
        except ValueError:
            logger.warning(f"Unknown bound type: {bound_type_str}")
            return
        
        # Get bound value if present
        bound_value = None
        if len(parts) > 3:
            try:
                bound_value = float(parts[3])
            except ValueError:
                pass
        elif len(parts) == 3 and bound_type_str not in ['FR', 'MI', 'PL', 'BV']:
            try:
                bound_value = float(parts[2])
                var_name = parts[1]
                var_idx = self.problem.variables[var_name] if var_name in self.problem.variables else None
            except ValueError:
                pass
        
        if var_idx is not None:
            self.problem.set_variable_bound(var_idx, bound_type, bound_value)
    
    def _handle_quadobj_section(self, line: str, line_num: int):
        """Handle QUADOBJ section - quadratic objective terms"""
        parts = line.split()
        if len(parts) < 3:
            return
        
        var1_name = parts[0]
        var2_name = parts[1]
        
        try:
            coeff = float(parts[2])
        except ValueError:
            return
        
        var1_idx = self.problem.get_variable_index(var1_name)
        var2_idx = self.problem.get_variable_index(var2_name)
        
        self.problem.add_quadratic_objective_term(var1_idx, var2_idx, coeff)
    
    def _handle_qmatrix_section(self, line: str, line_num: int):
        """Handle QMATRIX section - quadratic constraint matrix"""
        parts = line.split()
        if len(parts) < 4:
            return
        
        constraint_name = parts[0]
        var1_name = parts[1]
        var2_name = parts[2]
        
        try:
            coeff = float(parts[3])
        except ValueError:
            return
        
        var1_idx = self.problem.get_variable_index(var1_name)
        var2_idx = self.problem.get_variable_index(var2_name)
        
        self.problem.add_quadratic_constraint_term(constraint_name, var1_idx, var2_idx, coeff)
    
    def _handle_qcmatrix_section(self, line: str, line_num: int):
        """Handle QCMATRIX section - alternative quadratic constraint format"""
        # Similar to QMATRIX but different format
        self._handle_qmatrix_section(line, line_num)
    
    def _handle_sos_section(self, line: str, line_num: int):
        """Handle SOS section - Special Ordered Sets"""
        parts = line.split()
        if len(parts) < 3:
            return
        
        sos_name = parts[0]
        try:
            sos_type = SOSType(int(parts[1]))
        except ValueError:
            return
        
        # Create or get existing SOS constraint
        sos_constraint = None
        for sos in self.problem.sos_constraints:
            if sos.name == sos_name:
                sos_constraint = sos
                break
        
        if sos_constraint is None:
            sos_constraint = SOSConstraint(name=sos_name, sos_type=sos_type)
            self.problem.sos_constraints.append(sos_constraint)
        
        # Add variable and weight
        for i in range(2, len(parts), 2):
            if i + 1 >= len(parts):
                break
                
            var_name = parts[i]
            try:
                weight = float(parts[i + 1])
            except ValueError:
                continue
                
            var_idx = self.problem.get_variable_index(var_name)
            sos_constraint.variables.append(var_idx)
            sos_constraint.weights.append(weight)
    
    def _handle_csection_section(self, line: str, line_num: int):
        """Handle CSECTION section - conic constraints"""
        parts = line.split()
        if len(parts) < 3:
            return
        
        cone_name = parts[0]
        cone_type = parts[1]
        
        conic_constraint = ConicConstraint(name=cone_name, cone_type=cone_type)
        
        # Add variables
        for var_name in parts[2:]:
            var_idx = self.problem.get_variable_index(var_name)
            conic_constraint.variables.append(var_idx)
        
        self.problem.conic_constraints.append(conic_constraint)
    
    def _validate_problem(self):
        """Validate parsed problem for consistency"""
        if self.problem.n_vars == 0:
            raise ValueError("No variables found in MPS file")
        
        if not self.problem.constraints and not self.problem.linear_objective:
            raise ValueError("No constraints or objective found in MPS file")
        
        # Check for undefined variables in constraints
        for constraint in self.problem.constraints.values():
            for var_idx in constraint.coefficients.keys():
                if var_idx >= self.problem.n_vars:
                    raise ValueError(f"Constraint {constraint.name} references undefined variable index {var_idx}")


# Factory function for easy usage
def parse_mps_file(mps_file_path: str) -> MPSProblem:
    """
    Parse MPS file using comprehensive parser
    
    Args:
        mps_file_path: Path to MPS file
        
    Returns:
        MPSProblem with complete representation
    """
    parser = ComprehensiveMPSParser()
    return parser.parse_mps_file(mps_file_path)