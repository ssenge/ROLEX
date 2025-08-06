"""
MPS to Pyomo Converter Utility

This module provides functionality to convert MPS files directly to Pyomo ConcreteModel objects.
It extracts and adapts the MPS parsing logic from the SciPy solver.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

try:
    import pyomo.environ as pyo
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    pyo = None

logger = logging.getLogger(__name__)


class MPSToPyomoConverter:
    """Converts MPS files directly to Pyomo ConcreteModel objects"""
    
    @staticmethod
    def parse_mps_to_pyomo(mps_file_path: str) -> Optional['pyo.ConcreteModel']:
        """
        Parse an MPS file and create a Pyomo ConcreteModel
        
        Args:
            mps_file_path: Path to the MPS file
            
        Returns:
            Pyomo ConcreteModel ready for solving, or None if parsing failed
            
        Raises:
            ImportError: If Pyomo is not available
            FileNotFoundError: If MPS file doesn't exist
            ValueError: If MPS file format is invalid
        """
        if not PYOMO_AVAILABLE:
            raise ImportError("Pyomo not available. Install with: pip install pyomo")
        
        logger.info(f"Parsing MPS file to Pyomo model: {mps_file_path}")
        
        # Parse MPS file to get structured data
        mps_data = MPSToPyomoConverter._parse_mps_file(mps_file_path)
        
        # Create Pyomo model from parsed data
        model = MPSToPyomoConverter._create_pyomo_model(mps_data)
        
        return model
    
    @staticmethod
    def _parse_mps_file(mps_file_path: str) -> Dict[str, Any]:
        """
        Parse MPS file and extract optimization problem data
        (Adapted from SciPy MPS solver)
        
        Returns:
            Dict with variables, constraints, objective, bounds
        """
        variables = {}  # var_name -> index
        var_names = []  # index -> var_name
        constraints = {}  # constraint_name -> {'type': 'L'/'G'/'E', 'coeffs': {var_idx: coeff}, 'rhs': value}
        objective = {}  # var_idx -> coeff
        bounds = {}  # var_idx -> (lower, upper)
        obj_name = None
        
        current_section = None
        
        try:
            with open(mps_file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
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
                    
                    try:
                        if current_section == 'ROWS':
                            row_type = parts[0]
                            row_name = parts[1]
                            if row_type in ['L', 'G', 'E']:
                                constraints[row_name] = {'type': row_type, 'coeffs': {}, 'rhs': 0.0}
                            elif row_type == 'N':
                                # Objective row (first N row is typically the objective)
                                if obj_name is None:
                                    obj_name = row_name
                        
                        elif current_section == 'COLUMNS':
                            if 'MARKER' in line:
                                # Skip integer markers for now
                                continue
                            
                            var_name = parts[0]
                            if var_name not in variables:
                                variables[var_name] = len(var_names)
                                var_names.append(var_name)
                                bounds[len(var_names) - 1] = (0, None)  # Default bounds: x >= 0
                            
                            var_idx = variables[var_name]
                            
                            # Process coefficient pairs
                            for i in range(1, len(parts), 2):
                                if i + 1 < len(parts):
                                    constraint_name = parts[i]
                                    coeff = float(parts[i + 1])
                                    
                                    if constraint_name in constraints:
                                        constraints[constraint_name]['coeffs'][var_idx] = coeff
                                    elif constraint_name == obj_name:
                                        # Objective coefficient
                                        objective[var_idx] = coeff
                        
                        elif current_section == 'RHS':
                            rhs_name = parts[1] if len(parts) > 1 else None
                            for i in range(2, len(parts), 2):
                                if i + 1 < len(parts):
                                    constraint_name = parts[i]
                                    rhs_value = float(parts[i + 1])
                                    if constraint_name in constraints:
                                        constraints[constraint_name]['rhs'] = rhs_value
                        
                        elif current_section == 'BOUNDS':
                            bound_type = parts[0]
                            bound_name = parts[1] if len(parts) > 2 else None
                            var_name = parts[2] if len(parts) > 2 else parts[1]
                            
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
                                elif bound_type == 'MI':  # Minus infinity (lower bound)
                                    lower, upper = bounds.get(var_idx, (0, None))
                                    bounds[var_idx] = (None, upper)
                                elif bound_type == 'PL':  # Plus infinity (upper bound)
                                    lower, upper = bounds.get(var_idx, (0, None))
                                    bounds[var_idx] = (lower, None)
                    
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing line {line_num}: {line}. Error: {e}")
                        continue
                        
        except FileNotFoundError:
            raise FileNotFoundError(f"MPS file not found: {mps_file_path}")
        except Exception as e:
            raise ValueError(f"Error reading MPS file {mps_file_path}: {str(e)}")
        
        if not variables:
            raise ValueError("No variables found in MPS file")
        
        return {
            'variables': variables,
            'var_names': var_names,
            'constraints': constraints,
            'objective': objective,
            'bounds': bounds,
            'obj_name': obj_name,
            'n_vars': len(var_names),
            'n_constraints': len(constraints)
        }
    
    @staticmethod
    def _create_pyomo_model(mps_data: Dict[str, Any]) -> 'pyo.ConcreteModel':
        """
        Create a Pyomo ConcreteModel from parsed MPS data
        
        Args:
            mps_data: Parsed MPS data from _parse_mps_file
            
        Returns:
            Pyomo ConcreteModel ready for solving
        """
        model = pyo.ConcreteModel()
        
        var_names = mps_data['var_names']
        variables = mps_data['variables']
        constraints = mps_data['constraints']
        objective = mps_data['objective']
        bounds = mps_data['bounds']
        n_vars = mps_data['n_vars']
        
        logger.info(f"Creating Pyomo model: {n_vars} variables, {len(constraints)} constraints")
        
        # Create variable indices
        model.var_indices = pyo.RangeSet(0, n_vars - 1)
        
        # Create variables with bounds
        def var_bounds(model, i):
            lower, upper = bounds.get(i, (0, None))
            return (lower, upper)
        
        model.x = pyo.Var(model.var_indices, bounds=var_bounds, within=pyo.Reals)
        
        # Store variable names for result extraction
        model.var_names = var_names
        
        # Create objective function
        if objective:
            def obj_rule(model):
                return sum(coeff * model.x[var_idx] for var_idx, coeff in objective.items())
            model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        else:
            # If no objective, create a dummy one
            model.objective = pyo.Objective(expr=0, sense=pyo.minimize)
        
        # Create constraints
        constraint_list = []
        for constraint_name, constraint_data in constraints.items():
            constraint_type = constraint_data['type']
            coeffs = constraint_data['coeffs']
            rhs = constraint_data['rhs']
            
            # Create constraint expression
            if coeffs:
                expr = sum(coeff * model.x[var_idx] for var_idx, coeff in coeffs.items())
                
                if constraint_type == 'L':  # <=
                    constraint_list.append(expr <= rhs)
                elif constraint_type == 'G':  # >=
                    constraint_list.append(expr >= rhs)
                elif constraint_type == 'E':  # =
                    constraint_list.append(expr == rhs)
        
        # Add all constraints to the model
        if constraint_list:
            model.constraints = pyo.ConstraintList()
            for constraint in constraint_list:
                model.constraints.add(constraint)
        
        logger.info(f"Pyomo model created successfully: {len(model.x)} variables, {len(model.constraints) if hasattr(model, 'constraints') else 0} constraints")
        
        return model


def get_model_info(model: 'pyo.ConcreteModel') -> Dict[str, Any]:
    """
    Get information about a Pyomo model
    
    Args:
        model: Pyomo ConcreteModel
        
    Returns:
        Dict with model information
    """
    if not PYOMO_AVAILABLE or model is None:
        return {}
    
    info = {
        'num_variables': len(model.x) if hasattr(model, 'x') else 0,
        'num_constraints': len(model.constraints) if hasattr(model, 'constraints') else 0,
        'has_objective': hasattr(model, 'objective'),
        'variable_names': getattr(model, 'var_names', [])
    }
    
    return info