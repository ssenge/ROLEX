"""
Output formatting for ROLEX CLI results.
"""

import json
import csv
import sys
from io import StringIO

def format_output(result, args):
    """Format and display results based on args."""
    
    if args.format == 'json':
        output_json(result, args)
    elif args.format == 'csv':
        output_csv(result, args)
    else:
        if args.quiet:
            format_quiet(result)
        elif args.verbose:
            format_verbose(result, args)
        else:
            format_standard(result, args)

def format_quiet(result):
    """Minimal output format."""
    if result.is_optimal():
        print(f"{result.objective_value}")
    else:
        print(f"Status: {result.status}")

def format_standard(result, args):
    """Standard output format."""
    print(f"✅ Job completed successfully!")
    print(f"Status: {result.status}")
    print(f"Solve time: {result.solve_time:.4f}s")
    print(f"Solver: {result.solver}")
    print(f"Objective: {result.objective_value}")
    
    if args.show_vars and result.get_variables():
        print(f"\nVariables:")
        for name, value in result.get_variables().items():
            print(f"  {name} = {value}")

def format_verbose(result, args):
    """Verbose output format."""
    print(f"🚀 ROLEX CLI - Optimization Results")
    print(f"=" * 50)
    print(f"📋 Problem: {len(result.problem.get_variable_mapping()) if result.problem else 'N/A'} variables")
    print(f"⚙️  Solver: {result.solver}")
    print(f"⏱️  Solve time: {result.solve_time:.4f}s")
    print(f"")
    print(f"📊 Results:")
    print(f"  Status: {result.status}")
    print(f"  Objective value: {result.objective_value}")
    
    if args.show_vars and result.get_variables():
        print(f"\nVariables:")
        for name, value in result.get_variables().items():
            print(f"  {name} = {value}")
    elif result.get_variables():
        print(f"  Variables: {', '.join(f'{k}={v}' for k, v in result.get_variables().items())}")

def output_json(result, args):
    """Output results in JSON format."""
    data = {
        "status": result.status,
        "objective_value": result.objective_value,
        "solve_time": result.solve_time,
        "solver": result.solver,
        "message": result.message,
        "variables": result.get_variables(),
        "is_optimal": result.is_optimal(),
        "is_feasible": result.is_feasible()
    }
    
    json_str = json.dumps(data, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(json_str)
        print(f"✅ Results written to {args.output}")
    else:
        print(json_str)

def output_csv(result, args):
    """Output results in CSV format."""
    output = StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['variable', 'value'])
    
    # Variables
    for name, value in result.get_variables().items():
        writer.writerow([name, value])
    
    csv_str = output.getvalue()
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(csv_str)
        print(f"✅ Results written to {args.output}")
    else:
        print(csv_str) 