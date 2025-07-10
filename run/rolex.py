#!/usr/bin/env python3
"""
ROLEX - Remote Optimization Library EXecution
Command-line interface for optimization problems.
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'client'))

import client
from client import Result
from cli.args import create_parser
from cli.python_loader import load_python_instance  
from cli.formatters import format_output
from cli.validators import validate_inputs
import requests

def poll_with_progress(client, job_id, problem, args):
    """Poll for job completion with visual progress indication."""
    start_time = time.time()
    attempt = 0
    
    while attempt < args.max_attempts:
        attempt += 1
        
        try:
            response = requests.get(f"{client.server_url}/jobs/{job_id}", timeout=client.timeout)
            
            if response.status_code == 200:
                job_status = response.json()
                status = job_status["status"]
                
                if status == "completed":
                    if not args.quiet:
                        # Clear progress line and show completion
                        print(f"\r✅ Job completed after {time.time() - start_time:.1f}s (attempt {attempt}/{args.max_attempts})")
                    return Result.from_server_response(job_status["result"], problem)
                elif status == "failed":
                    error_msg = job_status.get("error", "Unknown error")
                    raise Exception(f"Job failed: {error_msg}")
                elif status in ["running", "queued"]:
                    # Show progress indication
                    if not args.quiet:
                        elapsed = time.time() - start_time
                        dots = "." * (attempt % 4)
                        print(f"\r⏳ Polling{dots:<3} [{attempt}/{args.max_attempts}] {elapsed:.1f}s", end="", flush=True)
                    
                    # Wait before next poll
                    time.sleep(args.poll_interval)
                    continue
                else:
                    raise Exception(f"Unknown job status: {status}")
            else:
                raise Exception(f"Failed to get job status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Could not connect to server: {e}")
    
    raise Exception(f"Timeout after {args.max_attempts} attempts")

def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Validation
        validate_inputs(args)
        
        # Load problem from input source
        if args.python:
            instance = load_python_instance(args)
        else:
            instance = client.Converter.from_file(args.input_file)
        
        # Create problem and client
        solver_params = {
            'time_limit': args.time_limit,
            'gap': args.gap,
            'verbose': args.verbose_solver
        }
        
        if args.threads:
            solver_params['threads'] = args.threads
        
        problem = client.Problem.from_instance(
            instance, 
            solver=args.solver,
            parameters=solver_params
        )
        
        rolex_client = client.Client(args.server_url, args.timeout)
        
        # Health check if requested
        if args.health_check:
            if not args.quiet:
                print(f"🔍 Checking server health...")
            if not rolex_client.health_check():
                print("❌ Server health check failed")
                sys.exit(1)
            if not args.quiet:
                print("✅ Server is healthy")
        
        # Submit job
        if not args.quiet:
            print(f"📤 Submitting job to {args.server_url}...")
        
        job_id = rolex_client.submit(problem)
        
        if not args.quiet:
            print(f"✅ Job submitted: {job_id}")
        
        if args.no_poll:
            return
        
        # Poll for results
        if not args.quiet:
            print(f"⏳ Polling for results...")
        
        result = poll_with_progress(rolex_client, job_id, problem, args)
        
        # Format and output results
        format_output(result, args)
        
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 