#!/usr/bin/env python3
"""
ROLEX CLI - Command Line Interface for MPS Files
Simple CLI to submit MPS files to ROLEX server and get results
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RolexMPSCLI:
    """CLI client for ROLEX MPS optimization"""
    
    def __init__(self, server_url: str = "http://localhost:80"):
        self.server_url = server_url.rstrip('/')
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def solve_mps(self, mps_file_path: str, solver: str = 'gurobi', 
                  max_time: int = 300, **kwargs) -> Dict[str, Any]:
        """
        Submit MPS file to ROLEX server and poll for results
        
        Args:
            mps_file_path: Path to MPS file
            solver: Solver to use ('gurobi' or 'cuopt')
            max_time: Maximum solve time in seconds
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with optimization results
        """
        # Validate file exists
        if not os.path.exists(mps_file_path):
            raise FileNotFoundError(f"MPS file not found: {mps_file_path}")
        
        # Check server availability
        if not self._check_server_health():
            raise RuntimeError(f"ROLEX server not available at {self.server_url}")
        
        # Submit job
        job_id = self._submit_mps_job(mps_file_path, solver, max_time, **kwargs)
        
        # Display initial status
        print(f"🚀 Job {job_id} submitted to {solver.upper()} solver")
        print(f"📁 File: {os.path.basename(mps_file_path)}")
        print(f"⏱️  Max time: {max_time}s")
        if kwargs:
            print(f"⚙️  Parameters: {kwargs}")
        print("📊 Polling for results...")
        
        # Poll for results
        result = self._poll_for_results(job_id)
        
        # Pretty print results
        self._print_results(result, solver, mps_file_path)
        
        return result
    
    def _check_server_health(self) -> bool:
        """Check if ROLEX server is available"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _submit_mps_job(self, mps_file_path: str, solver: str, 
                       max_time: int, **kwargs) -> str:
        """Submit MPS file to server"""
        # Build parameters
        parameters = {'max_time': max_time, **kwargs}
        
        try:
            with open(mps_file_path, 'rb') as f:
                files = {'mps_file': f}
                data = {
                    'solver': solver,
                    'parameters': json.dumps(parameters)
                }
                
                response = self.session.post(
                    f"{self.server_url}/jobs/submit-mps",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            response.raise_for_status()
            return response.json()['job_id']
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to submit job: {str(e)}")
        except KeyError:
            raise RuntimeError("Invalid server response: missing job_id")
    
    def _poll_for_results(self, job_id: str) -> Dict[str, Any]:
        """Poll server for job results"""
        poll_count = 0
        
        while True:
            try:
                response = self.session.get(
                    f"{self.server_url}/jobs/{job_id}/mps",
                    timeout=10
                )
                response.raise_for_status()
                result = response.json()
                
                status = result.get('status', 'unknown')
                
                # Show progress
                poll_count += 1
                if poll_count % 10 == 0:  # Show progress every 10 polls
                    print(f"📊 Still polling... ({poll_count}s)")
                
                # Check if job is complete
                if status in ['completed', 'failed']:
                    return result
                
                # Wait before next poll
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Polling error: {str(e)}")
                time.sleep(5)  # Wait longer on error
            except KeyboardInterrupt:
                print("\n❌ Interrupted by user")
                sys.exit(1)
    
    def _print_results(self, result: Dict[str, Any], solver: str, mps_file_path: str):
        """Pretty print optimization results"""
        print(f"\n{'='*60}")
        print(f"🎯 ROLEX MPS OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        
        # Basic info
        print(f"📁 File: {os.path.basename(mps_file_path)}")
        print(f"🔧 Solver: {solver.upper()}")
        print(f"🆔 Job ID: {result['job_id']}")
        
        # Status
        status = result.get('status', 'unknown')
        status_emoji = self._get_status_emoji(status)
        print(f"📈 Status: {status_emoji} {status.upper()}")
        
        # Check if job failed
        if status == 'failed':
            error = result.get('error', 'Unknown error')
            print(f"❌ Error: {error}")
            return
        
        # Get optimization result
        optimization_result = result.get('result')
        if not optimization_result:
            print("❌ No optimization result available")
            return
        
        # Objective value
        obj_value = optimization_result.get('objective_value')
        if obj_value is not None:
            print(f"🎯 Objective Value: {obj_value}")
        
        # Timing information
        solve_time = optimization_result.get('solve_time')
        total_time = optimization_result.get('total_time')
        if solve_time is not None:
            print(f"⏱️  Solve Time: {solve_time:.4f}s")
        if total_time is not None:
            print(f"🕐 Total Time: {total_time:.4f}s")
        
        # Solver info
        solver_info = optimization_result.get('solver_info', {})
        if solver_info.get('iterations'):
            print(f"🔄 Iterations: {solver_info['iterations']}")
        if solver_info.get('nodes'):
            print(f"🌳 Nodes: {solver_info['nodes']}")
        if solver_info.get('gap') is not None:
            print(f"🎯 Gap: {solver_info['gap']:.6f}")
        
        # Variable values
        variables = optimization_result.get('variables', {})
        if variables:
            print(f"\n📊 VARIABLE VALUES ({len(variables)} variables):")
            # Show up to 20 variables, sorted by name
            sorted_vars = sorted(variables.items())
            for i, (var_name, var_value) in enumerate(sorted_vars[:20]):
                print(f"  {var_name} = {var_value}")
            
            if len(variables) > 20:
                print(f"  ... and {len(variables) - 20} more variables")
        
        # Parameters used
        params = optimization_result.get('parameters_used', {})
        if params:
            print(f"\n⚙️  PARAMETERS USED:")
            for param, value in params.items():
                print(f"  {param}: {value}")
        
        # Timing summary
        if solve_time is not None:
            submitted_at = result.get('submitted_at', '')
            started_at = result.get('started_at', '')
            completed_at = result.get('completed_at', '')
            
            print(f"\n⏰ TIMING SUMMARY:")
            print(f"  Submitted: {submitted_at}")
            print(f"  Started: {started_at}")
            print(f"  Completed: {completed_at}")
        
        print(f"\n{'='*60}")
    
    def _get_status_emoji(self, status: str) -> str:
        """Get emoji for status"""
        status_emojis = {
            'optimal': '✅',
            'completed': '✅',
            'infeasible': '❌',
            'unbounded': '♾️',
            'time_limit': '⏰',
            'failed': '❌',
            'unknown': '❓'
        }
        return status_emojis.get(status, '❓')
    
    def list_solvers(self) -> Dict[str, Any]:
        """List available MPS solvers"""
        try:
            response = self.session.get(f"{self.server_url}/solvers/mps", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get solvers: {str(e)}")
    
    def print_solvers(self):
        """Print available solvers in a nice format"""
        try:
            solvers = self.list_solvers()
            
            print("🔧 Available MPS Solvers:")
            print("=" * 40)
            
            for solver_name, solver_info in solvers.items():
                available = solver_info.get('available', False)
                status_emoji = '✅' if available else '❌'
                print(f"{status_emoji} {solver_name.upper()}")
                
                if available:
                    capabilities = solver_info.get('capabilities', [])
                    if capabilities:
                        print(f"   Capabilities: {', '.join(capabilities)}")
                    
                    version = solver_info.get('version')
                    if version:
                        print(f"   Version: {version}")
                else:
                    status = solver_info.get('status', 'unknown')
                    print(f"   Status: {status}")
                
                print()
                
        except Exception as e:
            print(f"❌ Error listing solvers: {str(e)}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='ROLEX MPS CLI - Submit MPS files to ROLEX server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s problem.mps                    # Solve with default Gurobi
  %(prog)s problem.mps --solver cuopt     # Use cuOpt solver
  %(prog)s problem.mps --max-time 60      # Set time limit
  %(prog)s problem.mps --threads 4        # Use 4 threads
  %(prog)s --list-solvers                 # Show available solvers
        """
    )
    
    # Main arguments
    parser.add_argument(
        'mps_file', 
        nargs='?',
        help='Path to MPS file'
    )
    
    # Solver options
    parser.add_argument(
        '--solver', 
        choices=['gurobi', 'cuopt'], 
        default='gurobi',
        help='Solver to use (default: gurobi)'
    )
    
    # Parameters
    parser.add_argument(
        '--max-time', 
        type=int, 
        default=300,
        help='Maximum solve time in seconds (default: 300)'
    )
    
    parser.add_argument(
        '--threads', 
        type=int,
        help='Number of threads to use'
    )
    
    parser.add_argument(
        '--gap', 
        type=float,
        help='Gap tolerance (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    # Server options
    parser.add_argument(
        '--server', 
        default='http://localhost:8000',
        help='ROLEX server URL (default: http://localhost:8000)'
    )
    
    # Actions
    parser.add_argument(
        '--list-solvers', 
        action='store_true',
        help='List available solvers and exit'
    )
    
    args = parser.parse_args()
    
    # Create CLI client
    cli = RolexMPSCLI(args.server)
    
    # Handle list solvers
    if args.list_solvers:
        cli.print_solvers()
        return
    
    # Validate MPS file argument
    if not args.mps_file:
        parser.error("MPS file is required (or use --list-solvers)")
    
    # Build parameters
    kwargs = {}
    if args.threads:
        kwargs['threads'] = args.threads
    if args.gap:
        kwargs['gap_tolerance'] = args.gap
    if args.verbose:
        kwargs['verbose'] = True
    
    # Solve
    try:
        cli.solve_mps(
            args.mps_file,
            solver=args.solver,
            max_time=args.max_time,
            **kwargs
        )
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 