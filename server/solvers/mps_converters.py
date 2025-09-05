"""
MPS Problem Converters

This module provides converters from the unified MPSProblem representation
to various solver-specific formats (CVXPY, Pyomo, etc.).
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union

from .comprehensive_mps_parser import MPSProblem, ObjectiveSense, ConstraintType

# Optional imports with availability flags
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None

try:
    import pyomo.environ as pyo
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    pyo = None

logger = logging.getLogger(__name__)


def to_cvxpy(mps_problem: MPSProblem) -> 'cp.Problem':
    """
    Convert MPSProblem to CVXPY problem
    
    Args:
        mps_problem: Parsed MPS problem
        
    Returns:
        CVXPY Problem ready for solving
        
    Raises:
        ImportError: If CVXPY is not available
        ValueError: If problem contains unsupported features
    """
    if not CVXPY_AVAILABLE:
        raise ImportError("CVXPY not available. Install with: pip install cvxpy")
    
    logger.info(f"Converting MPSProblem to CVXPY: {mps_problem.n_vars} variables, "
               f"{len(mps_problem.constraints)} constraints")
    
    # Create variables
    x = cp.Variable(mps_problem.n_vars)
    constraints = []
    
    # Variable bounds
    for var_idx, (lb, ub) in mps_problem.var_bounds.items():
        if lb != -float('inf') and not np.isinf(lb):
            constraints.append(x[var_idx] >= lb)
        if ub != float('inf') and not np.isinf(ub):
            constraints.append(x[var_idx] <= ub)
    
    # Binary variables (CVXPY handles this specially)
    if mps_problem.binary_vars:
        logger.warning("Binary variables detected - CVXPY may require special handling for MIP")
        for var_idx in mps_problem.binary_vars:
            constraints.append(x[var_idx] >= 0)
            constraints.append(x[var_idx] <= 1)
            # Note: True integer constraints require special CVXPY setup
    
    # Integer variables
    if mps_problem.integer_vars:
        logger.warning("Integer variables detected - CVXPY may require special handling for MIP")
        # CVXPY integer handling would require cp.Variable(..., integer=True)
        # For now, we treat as continuous with bounds
    
    # Linear objective
    obj_expr = 0
    for var_idx, coeff in mps_problem.linear_objective.items():
        obj_expr += coeff * x[var_idx]
    
    # Quadratic objective
    if mps_problem.quadratic_objective:
        logger.info(f"Adding {len(mps_problem.quadratic_objective)} quadratic objective terms")
        
        # Build Q matrix for quadratic form: (1/2) * x^T * Q * x
        Q = np.zeros((mps_problem.n_vars, mps_problem.n_vars))
        
        for (i, j), coeff in mps_problem.quadratic_objective.items():
            # MPS format uses full coefficient, CVXPY quad_form uses 1/2 x^T Q x
            if i == j:
                Q[i, j] = coeff  # Diagonal terms
            else:
                # Off-diagonal terms: split coefficient between Q[i,j] and Q[j,i]
                Q[i, j] = coeff / 2
                Q[j, i] = coeff / 2
        
        obj_expr += cp.quad_form(x, Q)
    
    # Linear constraints
    for constraint in mps_problem.constraints.values():
        if not constraint.coefficients:
            continue
            
        # Build constraint expression
        lhs = sum(coeff * x[var_idx] for var_idx, coeff in constraint.coefficients.items())
        
        if constraint.constraint_type == ConstraintType.EQUAL:
            constraints.append(lhs == constraint.rhs)
        elif constraint.constraint_type == ConstraintType.LESS_EQUAL:
            constraints.append(lhs <= constraint.rhs)
        elif constraint.constraint_type == ConstraintType.GREATER_EQUAL:
            constraints.append(lhs >= constraint.rhs)
        
        # Handle range constraints
        if constraint.range_value is not None:
            if constraint.constraint_type == ConstraintType.LESS_EQUAL:
                # L constraint with range: rhs - |range| <= lhs <= rhs
                constraints.append(lhs >= constraint.rhs - abs(constraint.range_value))
            elif constraint.constraint_type == ConstraintType.GREATER_EQUAL:
                # G constraint with range: rhs <= lhs <= rhs + |range|
                constraints.append(lhs <= constraint.rhs + abs(constraint.range_value))
            elif constraint.constraint_type == ConstraintType.EQUAL:
                # E constraint with range: rhs - |range| <= lhs <= rhs + |range|
                constraints.append(lhs >= constraint.rhs - abs(constraint.range_value))
                constraints.append(lhs <= constraint.rhs + abs(constraint.range_value))
    
    # Quadratic constraints
    if mps_problem.quadratic_constraints:
        logger.info(f"Adding {len(mps_problem.quadratic_constraints)} quadratic constraints")
        
        for constraint_name, quad_terms in mps_problem.quadratic_constraints.items():
            if constraint_name not in mps_problem.constraints:
                logger.warning(f"Quadratic constraint {constraint_name} not found in linear constraints")
                continue
            
            constraint = mps_problem.constraints[constraint_name]
            
            # Linear part
            lhs = sum(coeff * x[var_idx] for var_idx, coeff in constraint.coefficients.items())
            
            # Quadratic part
            Q = np.zeros((mps_problem.n_vars, mps_problem.n_vars))
            for (i, j), coeff in quad_terms.items():
                if i == j:
                    Q[i, j] = coeff
                else:
                    Q[i, j] = coeff / 2
                    Q[j, i] = coeff / 2
            
            quad_expr = cp.quad_form(x, Q)
            total_lhs = lhs + quad_expr
            
            if constraint.constraint_type == ConstraintType.EQUAL:
                constraints.append(total_lhs == constraint.rhs)
            elif constraint.constraint_type == ConstraintType.LESS_EQUAL:
                constraints.append(total_lhs <= constraint.rhs)
            elif constraint.constraint_type == ConstraintType.GREATER_EQUAL:
                constraints.append(total_lhs >= constraint.rhs)
    
    # SOS constraints
    if mps_problem.sos_constraints:
        logger.warning(f"SOS constraints not directly supported in CVXPY - {len(mps_problem.sos_constraints)} constraints ignored")
    
    # Conic constraints
    if mps_problem.conic_constraints:
        logger.info(f"Adding {len(mps_problem.conic_constraints)} conic constraints")
        
        for conic in mps_problem.conic_constraints:
            if conic.cone_type == "QUAD":
                # Second-order cone: ||x[1:]||_2 <= x[0]
                if len(conic.variables) >= 2:
                    x_cone = cp.vstack([x[i] for i in conic.variables])
                    constraints.append(cp.SOC(x_cone[0], x_cone[1:]))
            else:
                logger.warning(f"Unsupported conic constraint type: {conic.cone_type}")
    
    # Create objective
    if mps_problem.objective_sense == ObjectiveSense.MINIMIZE:
        objective = cp.Minimize(obj_expr)
    else:
        objective = cp.Maximize(obj_expr)
    
    problem = cp.Problem(objective, constraints)
    
    logger.info(f"CVXPY problem created: {len(constraints)} constraints")
    
    return problem


def to_pyomo(mps_problem: MPSProblem) -> 'pyo.ConcreteModel':
    """
    Convert MPSProblem to Pyomo ConcreteModel
    
    Args:
        mps_problem: Parsed MPS problem
        
    Returns:
        Pyomo ConcreteModel ready for solving
        
    Raises:
        ImportError: If Pyomo is not available
    """
    if not PYOMO_AVAILABLE:
        raise ImportError("Pyomo not available. Install with: pip install pyomo")
    
    logger.info(f"Converting MPSProblem to Pyomo: {mps_problem.n_vars} variables, "
               f"{len(mps_problem.constraints)} constraints")
    
    model = pyo.ConcreteModel()
    model.name = mps_problem.name or "MPS_Problem"
    
    # Variable indices
    model.var_indices = pyo.RangeSet(0, mps_problem.n_vars - 1)
    
    # Variables with bounds
    def var_bounds_rule(model, i):
        if i in mps_problem.var_bounds:
            lb, ub = mps_problem.var_bounds[i]
            # Convert inf to None for Pyomo
            lb = None if lb == -float('inf') else lb
            ub = None if ub == float('inf') else ub
            return (lb, ub)
        return (0, None)  # Default bounds
    
    # Determine variable domain
    def var_domain_rule(i):
        if i in mps_problem.binary_vars:
            return pyo.Binary
        elif i in mps_problem.integer_vars:
            return pyo.Integers
        else:
            return pyo.Reals
    
    model.x = pyo.Var(model.var_indices, bounds=var_bounds_rule, 
                     domain=lambda model, i: var_domain_rule(i))
    
    # Store variable names for reference
    model.var_names = mps_problem.var_names
    
    # Linear objective
    def obj_rule(model):
        expr = 0
        
        # Linear terms
        for var_idx, coeff in mps_problem.linear_objective.items():
            expr += coeff * model.x[var_idx]
        
        # Quadratic terms
        for (i, j), coeff in mps_problem.quadratic_objective.items():
            expr += coeff * model.x[i] * model.x[j]
        
        return expr
    
    if mps_problem.objective_sense == ObjectiveSense.MINIMIZE:
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    else:
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
    
    # Linear constraints
    model.constraints = pyo.ConstraintList()
    
    for constraint in mps_problem.constraints.values():
        if not constraint.coefficients:
            continue
        
        # Linear part
        linear_expr = sum(coeff * model.x[var_idx] 
                         for var_idx, coeff in constraint.coefficients.items())
        
        # Add quadratic part if exists
        if constraint.name in mps_problem.quadratic_constraints:
            quad_terms = mps_problem.quadratic_constraints[constraint.name]
            for (i, j), coeff in quad_terms.items():
                linear_expr += coeff * model.x[i] * model.x[j]
        
        # Add constraint based on type
        if constraint.constraint_type == ConstraintType.EQUAL:
            model.constraints.add(linear_expr == constraint.rhs)
        elif constraint.constraint_type == ConstraintType.LESS_EQUAL:
            model.constraints.add(linear_expr <= constraint.rhs)
        elif constraint.constraint_type == ConstraintType.GREATER_EQUAL:
            model.constraints.add(linear_expr >= constraint.rhs)
        
        # Handle range constraints
        if constraint.range_value is not None:
            range_val = abs(constraint.range_value)
            
            if constraint.constraint_type == ConstraintType.LESS_EQUAL:
                # L constraint with range: rhs - range <= lhs <= rhs
                model.constraints.add(linear_expr >= constraint.rhs - range_val)
            elif constraint.constraint_type == ConstraintType.GREATER_EQUAL:
                # G constraint with range: rhs <= lhs <= rhs + range
                model.constraints.add(linear_expr <= constraint.rhs + range_val)
            elif constraint.constraint_type == ConstraintType.EQUAL:
                # E constraint with range: rhs - range <= lhs <= rhs + range
                model.constraints.add(linear_expr >= constraint.rhs - range_val)
                model.constraints.add(linear_expr <= constraint.rhs + range_val)
    
    # SOS constraints
    if mps_problem.sos_constraints:
        logger.info(f"Adding {len(mps_problem.sos_constraints)} SOS constraints")
        model.sos_constraints = pyo.SOSConstraintList()
        
        for sos in mps_problem.sos_constraints:
            sos_vars = [model.x[var_idx] for var_idx in sos.variables]
            sos_weights = sos.weights or list(range(1, len(sos.variables) + 1))
            
            if sos.sos_type.value == 1:
                model.sos_constraints.add(pyo.SOS1(sos_vars, sos_weights))
            elif sos.sos_type.value == 2:
                model.sos_constraints.add(pyo.SOS2(sos_vars, sos_weights))
    
    # Conic constraints - limited support in Pyomo
    if mps_problem.conic_constraints:
        logger.warning(f"Conic constraints have limited support in Pyomo - "
                      f"{len(mps_problem.conic_constraints)} constraints may not be handled correctly")
    
    logger.info(f"Pyomo model created: {len(model.x)} variables, "
               f"{len(model.constraints) if hasattr(model, 'constraints') else 0} constraints, "
               f"{len(model.sos_constraints) if hasattr(model, 'sos_constraints') else 0} SOS constraints")
    
    return model


def to_scipy_matrices(mps_problem: MPSProblem) -> Dict[str, Any]:
    """
    Convert MPSProblem to SciPy-compatible matrix format
    
    Args:
        mps_problem: Parsed MPS problem
        
    Returns:
        Dict with c, A_ub, b_ub, A_eq, b_eq, bounds for scipy.optimize.linprog
        
    Note:
        Only supports linear problems - quadratic terms are ignored
    """
    logger.info(f"Converting MPSProblem to SciPy matrices: {mps_problem.n_vars} variables")
    
    if mps_problem.quadratic_objective or mps_problem.quadratic_constraints:
        logger.warning("Quadratic terms detected but not supported in SciPy conversion - ignoring")
    
    # Objective vector (c)
    c = np.zeros(mps_problem.n_vars)
    for var_idx, coeff in mps_problem.linear_objective.items():
        c[var_idx] = coeff
    
    # Separate equality and inequality constraints
    eq_constraints = []
    ineq_constraints = []
    
    for constraint in mps_problem.constraints.values():
        if not constraint.coefficients:
            continue
        
        # Build constraint row
        row = np.zeros(mps_problem.n_vars)
        for var_idx, coeff in constraint.coefficients.items():
            row[var_idx] = coeff
        
        if constraint.constraint_type == ConstraintType.EQUAL:
            eq_constraints.append((row, constraint.rhs))
        elif constraint.constraint_type == ConstraintType.LESS_EQUAL:
            ineq_constraints.append((row, constraint.rhs))
        elif constraint.constraint_type == ConstraintType.GREATER_EQUAL:
            # Convert >= to <= by negating
            ineq_constraints.append((-row, -constraint.rhs))
    
    # Build matrices
    A_eq = np.array([row for row, rhs in eq_constraints]) if eq_constraints else None
    b_eq = np.array([rhs for row, rhs in eq_constraints]) if eq_constraints else None
    
    A_ub = np.array([row for row, rhs in ineq_constraints]) if ineq_constraints else None
    b_ub = np.array([rhs for row, rhs in ineq_constraints]) if ineq_constraints else None
    
    # Variable bounds
    bounds = []
    for i in range(mps_problem.n_vars):
        if i in mps_problem.var_bounds:
            lb, ub = mps_problem.var_bounds[i]
            lb = None if lb == -float('inf') else lb
            ub = None if ub == float('inf') else ub
            bounds.append((lb, ub))
        else:
            bounds.append((0, None))  # Default bounds
    
    # Flip objective for maximization (scipy minimizes)
    if mps_problem.objective_sense == ObjectiveSense.MAXIMIZE:
        c = -c
    
    result = {
        'c': c,
        'A_ub': A_ub,
        'b_ub': b_ub,
        'A_eq': A_eq,
        'b_eq': b_eq,
        'bounds': bounds,
        'method': 'highs',  # Default SciPy method
        'maximize': mps_problem.objective_sense == ObjectiveSense.MAXIMIZE
    }
    
    logger.info(f"SciPy matrices created: {len(eq_constraints)} equality, "
               f"{len(ineq_constraints)} inequality constraints")
    
    return result


# Utility functions
def get_problem_info(mps_problem: MPSProblem) -> Dict[str, Any]:
    """Get summary information about MPS problem"""
    return {
        'name': mps_problem.name,
        'n_vars': mps_problem.n_vars,
        'n_constraints': len(mps_problem.constraints),
        'n_binary_vars': len(mps_problem.binary_vars),
        'n_integer_vars': len(mps_problem.integer_vars),
        'n_quadratic_obj_terms': len(mps_problem.quadratic_objective),
        'n_quadratic_constraints': len(mps_problem.quadratic_constraints),
        'n_sos_constraints': len(mps_problem.sos_constraints),
        'n_conic_constraints': len(mps_problem.conic_constraints),
        'objective_sense': mps_problem.objective_sense.value,
        'has_quadratic': bool(mps_problem.quadratic_objective or mps_problem.quadratic_constraints),
        'has_integer': bool(mps_problem.integer_vars or mps_problem.binary_vars),
        'has_sos': bool(mps_problem.sos_constraints),
        'has_conic': bool(mps_problem.conic_constraints)
    }