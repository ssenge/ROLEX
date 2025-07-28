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
import csv
import gzip
from typing import Dict, Any, Optional, List, Generator
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


class RolexMPSCLI:
    """CLI client for ROLEX MPS optimization"""

    def __init__(self, server_url: str = "http://localhost:80"):
        self.server_url = server_url.rstrip('/')
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy"""
        session = requests.Session()
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
                  max_time: int = 300, timeout: int = 600, **kwargs) -> Dict[str, Any]:
        """
        Submit MPS file to ROLEX server and poll for results

        Args:
            mps_file_path: Path to MPS file
            solver: Solver to use ('gurobi' or 'cuopt')
            max_time: Maximum solve time in seconds
            timeout: Upload timeout in seconds
            **kwargs: Additional parameters

        Returns:
            Dictionary with optimization results
        """
        if not os.path.exists(mps_file_path):
            raise FileNotFoundError(f"MPS file not found: {mps_file_path}")

        print("Checking server health...")
        if not self._check_server_health():
            raise RuntimeError(f"ROLEX server not available at {self.server_url}")

        job_id = self._submit_mps_job(mps_file_path, solver, max_time, timeout, **kwargs)

        print(f"\n🚀 Job {job_id} submitted to {solver.upper()} solver for file {os.path.basename(mps_file_path)}")
        
        result = self._poll_for_results(job_id)
        
        return result

    def _check_server_health(self) -> bool:
        """Check if ROLEX server is available"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _compress_stream_with_progress(self, file_path: str, chunk_size: int = 16384) -> Generator[bytes, None, None]:
        """A generator that reads a file, shows a progress bar, and yields compressed chunks."""
        file_size = os.path.getsize(file_path)
        
        with open(file_path, 'rb') as f, tqdm(
            desc="Uploading",
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                bar.update(len(chunk))
                yield gzip.compress(chunk)

    def _submit_mps_job(self, mps_file_path: str, solver: str,
                       max_time: int, timeout: int, **kwargs) -> str:
        """
        Compress and stream MPS file to server with progress bar."""
        parameters = {'max_time': max_time, **kwargs}
        
        data_stream = self._compress_stream_with_progress(mps_file_path)

        try:
            headers = {
                'Content-Encoding': 'gzip',
                'Content-Type': 'application/octet-stream'
            }
            params = {
                'solver': solver,
                'parameters': json.dumps(parameters),
                'filename': os.path.basename(mps_file_path) + ".gz"
            }
            
            response = self.session.post(
                f"{self.server_url}/jobs/submit-mps",
                params=params,
                data=data_stream,
                headers=headers,
                timeout=timeout
            )
            
            print("\nFile uploaded. Waiting for server to issue Job ID...")
            response.raise_for_status()
            return response.json()['job_id']
        except KeyError:
            raise RuntimeError("Invalid server response: missing job_id")
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Request timed out after {timeout} seconds. For larger files, consider increasing the timeout with the --timeout flag.")

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

                if poll_count % 10 == 0:
                    print(f"Polling for job {job_id}... status: {status}")

                if status in ['completed', 'failed']:
                    return result
                
                poll_count += 1
                time.sleep(5)
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Polling error for job {job_id}: {str(e)}")
                time.sleep(5)
            except KeyboardInterrupt:
                print(f"\n❌ Interrupted by user. Job {job_id} may still be running on the server.")
                sys.exit(1)

    def write_result_to_csv(self, result: Dict[str, Any], csv_path: str, input_filename: str, submission_timestamp: str, completion_timestamp: str, num_variables: int, solver_name: str, store_vars: bool = False):
        """Writes a single result to the specified CSV file."""
        
        header = ['input_filename', 'job_id', 'solver_engine', 'solver_status', 'objective_value', 'time_to_solution', 'num_variables', 'submission_timestamp', 'completion_timestamp', 'variable_assignments']
        
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            
            if not file_exists:
                writer.writeheader()

            opt_result = result.get('result', {})
            
            variable_assignments = ""
            if store_vars:
                variable_assignments = ";".join([f"{k}={v}" for k, v in opt_result.get('variables', {}).items()])

            row = {
                'input_filename': input_filename,
                'job_id': result.get('job_id'),
                'solver_engine': solver_name,
                'solver_status': result.get('status'),
                'objective_value': opt_result.get('objective_value'),
                'time_to_solution': opt_result.get('solve_time'),
                'num_variables': num_variables,
                'submission_timestamp': submission_timestamp,
                'completion_timestamp': completion_timestamp,
                'variable_assignments': variable_assignments
            }
            writer.writerow(row)

    def print_pretty_results(self, result: Dict[str, Any], solver: str, mps_file_path: str, show_vars: bool = False):
        """Pretty print optimization results"""
        print(f"\n{'='*60}")
        print(f"🎯 ROLEX MPS OPTIMIZATION RESULTS")
        print(f"\n{'='*60}")
        print(f"📁 File: {os.path.basename(mps_file_path)}")
        print(f"🔧 Solver: {solver.upper()}")
        print(f"🆔 Job ID: {result['job_id']}")
        status = result.get('status', 'unknown')
        status_emoji = self._get_status_emoji(status)
        print(f"📈 Status: {status_emoji} {status.upper()}")
        if status == 'failed':
            error = result.get('error', 'Unknown error')
            print(f"❌ Error: {error}")
            return
        optimization_result = result.get('result')
        if not optimization_result:
            print("❌ No optimization result available")
            return
        obj_value = optimization_result.get('objective_value')
        if obj_value is not None:
            print(f"🎯 Objective Value: {obj_value}")
        solve_time = optimization_result.get('solve_time')
        if solve_time is not None:
            print(f"⏱️  Solve Time: {solve_time:.4f}s")
        
        if show_vars:
            variables = optimization_result.get('variables', {})
            if variables:
                print(f"\n📊 VARIABLE VALUES ({len(variables)} variables):")
                sorted_vars = sorted(variables.items())
                for i, (var_name, var_value) in enumerate(sorted_vars[:20]):
                    print(f"  {var_name} = {var_value}")
                if len(variables) > 20:
                    print(f"  ... and {len(variables) - 20} more variables")
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
                    print(f"   Version: {solver_info.get('version')}")
                else:
                    print(f"   Status: {solver_info.get('status', 'unknown')}")
                print()
        except Exception as e:
            print(f"❌ Error listing solvers: {str(e)}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='ROLEX MPS CLI - Submit one or more MPS files to ROLEX server sequentially.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s problem1.mps problem2.mps --output-csv results.csv
  %(prog)s *.mps --solver cuopt --max-time 60 --output-csv results.csv
  %(prog)s problem.mps --threads 4
  %(prog)s problem.mps --output-csv results.csv --store-var-assignments
  %(prog)s problem.mps --show-var-assignments
  %(prog)s --list-solvers
  %(prog)s large_problem.mps --timeout 600
        """
    )

    parser.add_argument(
        'mps_files',
        nargs='*', 
        help='One or more paths to MPS files to be processed.'
    )
    parser.add_argument(
        '--output-csv',
        help='Path to CSV file to store results. If the file exists, results are appended.'
    )
    parser.add_argument(
        '--store-var-assignments',
        action='store_true',
        help='Store variable assignments in the CSV file. Can result in large files.'
    )
    parser.add_argument(
        '--show-var-assignments',
        action='store_true',
        help='Display variable assignments in the console output.'
    )
    parser.add_argument(
        '--solver',
        choices=['gurobi', 'cuopt'],
        default='gurobi',
        help='Solver to use (default: gurobi)'
    )
    parser.add_argument(
        '--max-time',
        type=int,
        default=300,
        help='Maximum solve time in seconds (default: 300)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='Upload timeout in seconds (default: 600)'
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
        help='Enable verbose output for each job.'
    )
    parser.add_argument(
        '--server',
        default='http://localhost:8000',
        help='ROLEX server URL (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--list-solvers',
        action='store_true',
        help='List available solvers and exit'
    )

    args = parser.parse_args()

    cli = RolexMPSCLI(args.server)

    if args.list_solvers:
        cli.print_solvers()
        return

    if not args.mps_files:
        parser.error("At least one MPS file is required, or use --list-solvers.")

    # Build parameters
    kwargs = {}
    if args.threads:
        kwargs['threads'] = args.threads
    if args.gap:
        kwargs['gap_tolerance'] = args.gap
    if args.verbose:
        kwargs['verbose'] = True

    total_files = len(args.mps_files)
    for i, mps_file in enumerate(args.mps_files):
        print(f"Processing file {i+1}/{total_files}: {mps_file}")
        try:
            submission_timestamp = datetime.now().isoformat()
            result = cli.solve_mps(
                mps_file,
                solver=args.solver,
                max_time=args.max_time,
                timeout=args.timeout,
                **kwargs
            )
            completion_timestamp = datetime.now().isoformat()

            if args.verbose:
                cli.print_pretty_results(result, args.solver, mps_file, show_vars=args.show_var_assignments)

            if args.output_csv:
                num_variables = len(result.get('result', {}).get('variables', {}))
                cli.write_result_to_csv(result, args.output_csv, os.path.basename(mps_file), submission_timestamp, completion_timestamp, num_variables, args.solver, store_vars=args.store_var_assignments)
                print(f"Results for {os.path.basename(mps_file)} written to {args.output_csv}")

        except Exception as e:
            print(f"❌ Error processing {mps_file}: {str(e)}")
            # Optionally write error to CSV as well
            if args.output_csv:
                error_result = {'job_id': 'failed_submission', 'status': 'error', 'error': str(e)}
                cli.write_result_to_csv(error_result, args.output_csv, os.path.basename(mps_file), submission_timestamp, datetime.now().isoformat(), 0, args.solver, store_vars=args.store_var_assignments)
        
        if i < total_files - 1:
            print("-" * 60)

if __name__ == "__main__":
    main()