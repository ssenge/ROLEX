"""
ROLEX Server - Gurobi MPS Solver
Native MPS file solving with Gurobi
"""
import time
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING, List

from .mps_base import BaseMPSSolver
from models import MPSOptimizationResponse, SolverDiagnostics, SolverCapability

# Import dependencies with proper error handling
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp = None
    GRB = None

# Type hints that work even when Gurobi is not available
if TYPE_CHECKING or GUROBI_AVAILABLE:
    from gurobipy import Model as GurobiModel
else:
    GurobiModel = object

logger = logging.getLogger(__name__)


class GurobiMPSSolver(BaseMPSSolver):
    """Gurobi solver with native MPS file support"""
    
    def __init__(self):
        super().__init__("Gurobi-MPS")
        
    def is_available(self) -> bool:
        """Check if Gurobi is available"""
        return GUROBI_AVAILABLE
    
    def get_capabilities(self) -> List[SolverCapability]:
        """Gurobi can solve both LP and MIP problems"""
        return [SolverCapability.LP, SolverCapability.MIP]
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get Gurobi solver information"""
        info = {
            "name": "Gurobi",
            "available": self.is_available(),
            "capabilities": [cap.value for cap in self.get_capabilities()]
        }
        
        # Add version if Gurobi is available
        if self.is_available() and GUROBI_AVAILABLE:
            try:
                import gurobipy as gp
                info["version"] = f"{gp.gurobi.version()[0]}.{gp.gurobi.version()[1]}"
            except Exception:
                info["version"] = "unknown"
        else:
            info["version"] = None
            
        return info
    
    def solve_mps(self, mps_file_path: str, parameters: Dict[str, Any], optimality_tolerance: Optional[float] = None) -> MPSOptimizationResponse:
        """
        Solve MPS file using Gurobi
        
        Args:
            mps_file_path: Path to MPS file
            parameters: Solver parameters
            
        Returns:
            MPSOptimizationResponse with solution
        """
        if not self.is_available():
            raise RuntimeError("Gurobi solver is not available")
        
        # Validate parameters
        validated_params = self._validate_parameters(parameters)
        
        
        try:
            # Read MPS file
            model = gp.read(mps_file_path)
            
            # Set parameters
            self._set_gurobi_parameters(model, validated_params, optimality_tolerance)
            
            # Data for convergence tracking
            convergence_data = []
            last_log_time = -float('inf')

            def _callback(model, where):
                nonlocal last_log_time
                log_frequency = validated_params.get('log_frequency')
                
                if not log_frequency:
                    return # No log frequency set, do nothing

                current_time = time.time() - start_time
                time_diff = current_time - last_log_time
                
                if time_diff >= log_frequency:
                    obj_val = None
                    if where == GRB.Callback.MIPNODE:
                        raw_obj_val = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
                        if raw_obj_val is not None:
                            obj_val = raw_obj_val
                    elif where == GRB.Callback.MIPSOL:
                        raw_obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        if raw_obj_val is not None:
                            obj_val = raw_obj_val
                    elif where == GRB.Callback.BARRIER:
                        raw_obj_val = model.cbGet(GRB.Callback.BARRIER_PRIMOBJ)
                        if raw_obj_val is not None:
                            obj_val = raw_obj_val
                    elif where == GRB.Callback.SIMPLEX:
                        raw_obj_val = model.cbGet(GRB.Callback.SPX_OBJVAL)
                        if raw_obj_val is not None:
                            obj_val = raw_obj_val
                    elif where == GRB.Callback.MESSAGE:
                        msg = model.cbGet(GRB.Callback.MSG_STRING)
                        if msg:
                            parts = msg.strip().split()
                            if len(parts) > 1:
                                try:
                                    # Assuming the second part is the objective value
                                    obj_val = float(parts[1])
                                except ValueError:
                                    # Log if parsing fails, but don't stop
                                    print(f"*** GUROBI LOG (where=6): Could not parse objective from: {msg.strip()}")
                            else:
                                print(f"*** GUROBI LOG (where=6): {msg.strip()}")

                    if obj_val is not None:
                        convergence_data.append({'time': current_time, 'objective': obj_val})
                        print(f"*** =============> Objective {obj_val} captured at time {current_time:.2f}s from where={where}")
                    
                    last_log_time = current_time # Always update last_log_time to ensure next log is correctly timed

            # Solve
            start_time = time.time()
            callback_to_use = None
            if validated_params.get('log_frequency') and validated_params['log_frequency'] > 0:
                callback_to_use = _callback
            model.optimize(callback=callback_to_use)
            solve_time = time.time() - start_time
            
            # Process results
            response = self._process_gurobi_results(model, solve_time, validated_params)
            print(f"=============> Convergence data: {convergence_data}")
            response.convergence_data = convergence_data
            return response
            
        except Exception as e:
            logger.error(f"Gurobi MPS solve failed: {str(e)}")
            raise RuntimeError(f"Gurobi solve failed: {str(e)}")
    
    def _set_gurobi_parameters(self, model: GurobiModel, parameters: Dict[str, Any], optimality_tolerance: Optional[float] = None):
        """Set Gurobi parameters from validated parameters"""
        
        if 'max_time' in parameters:
            model.setParam('TimeLimit', parameters['max_time'])
            logger.info(f"Set TimeLimit to {parameters['max_time']} seconds")
        
        if 'threads' in parameters:
            model.setParam('Threads', parameters['threads'])
            logger.info(f"Set Threads to {parameters['threads']}")
        
        
        
        if 'verbose' in parameters:
            if parameters['verbose']:
                model.setParam('OutputFlag', 1)
            else:
                model.setParam('OutputFlag', 0)
        
        if optimality_tolerance is not None:
            if model.IsMIP:
                model.setParam('MIPGap', optimality_tolerance)
                logger.info(f"Set MIPGap to {optimality_tolerance} (from optimality_tolerance)")
            else:
                model.setParam('OptimalityTol', optimality_tolerance)
                logger.info(f"Set OptimalityTol to {optimality_tolerance} (from optimality_tolerance)")
    
    def _process_gurobi_results(self, model: GurobiModel, solve_time: float, parameters: Dict[str, Any]) -> MPSOptimizationResponse:
        """Process Gurobi optimization results"""
        
        # Map Gurobi status to our status
        status_map = {
            GRB.OPTIMAL: "optimal",
            GRB.INFEASIBLE: "infeasible",
            GRB.UNBOUNDED: "unbounded",
            GRB.INF_OR_UNBD: "infeasible_or_unbounded",
            GRB.TIME_LIMIT: "time_limit",
            GRB.NODE_LIMIT: "node_limit",
            GRB.SOLUTION_LIMIT: "solution_limit",
            GRB.INTERRUPTED: "interrupted",
            GRB.NUMERIC: "numeric_error",
            GRB.SUBOPTIMAL: "suboptimal"
        }
        
        status = status_map.get(model.Status, "unknown")
        
        # Get objective value
        objective_value = None
        if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
            try:
                objective_value = model.objVal
            except:
                objective_value = None
        
        # Get variables
        variables = {}
        if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
            try:
                for var in model.getVars():
                    variables[var.varName] = var.x
            except Exception as e:
                logger.warning(f"Could not extract variables: {str(e)}")
        
        # Get solver diagnostics
        solver_info = SolverDiagnostics()
        try:
            solver_info.iterations = int(model.getAttr('IterCount'))
        except:
            pass
        
        try:
            solver_info.nodes = int(model.getAttr('NodeCount'))
        except:
            pass
        
        try:
            solver_info.gap = float(model.getAttr('MIPGap'))
        except:
            pass
        
        try:
            solver_info.bound = float(model.getAttr('ObjBound'))
        except:
            pass
        
        solver_info.status_code = model.Status
        
        # Generate message
        message = f"Gurobi status: {status}"
        if objective_value is not None:
            message += f", objective: {objective_value}"
        
        return MPSOptimizationResponse(
            status=status,
            objective_value=objective_value,
            variables=variables,
            solve_time=solve_time,
            solver=self.name,
            message=message,
            solver_info=solver_info,
            num_constraints=model.NumConstrs,
            parameters_used=parameters
        )
    
    def _validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Gurobi-specific parameters"""
        validated = super()._validate_parameters(parameters)
        
        # Add Gurobi-specific parameter validation
        if 'feasibility_tolerance' in parameters:
            tol = parameters['feasibility_tolerance']
            if isinstance(tol, (int, float)) and tol > 0:
                validated['feasibility_tolerance'] = float(tol)
        
        if 'optimality_tolerance' in parameters:
            tol = parameters['optimality_tolerance']
            if isinstance(tol, (int, float)) and tol > 0:
                validated['optimality_tolerance'] = float(tol)
        
        return validated 