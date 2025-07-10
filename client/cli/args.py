"""
Command-line argument parsing for ROLEX CLI.
"""

import argparse

def create_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog='rolex',
        description='ROLEX - Remote Optimization Library EXecution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rolex problem.lp
  rolex --solver gurobi --time-limit 600 problem.mps
  rolex --python examples/cli_examples/simple_problem.py:create_instance
  rolex --python "examples/cli_examples/advanced_problem.py:tsp(cities=50)" --verbose --show-vars
  rolex -q --format json --output results.json problem.lp
        """
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        'input_file', 
        nargs='?',
        help='Optimization file (LP, MPS, or QPLIB format)'
    )
    input_group.add_argument(
        '--python',
        metavar='MODULE:FUNCTION',
        help='Python module and function (e.g., "module.py:create_instance")'
    )
    
    # Server & Connection
    server_group = parser.add_argument_group('Server Configuration')
    server_group.add_argument('--server-url', default='http://localhost:8000',
                             help='Server URL (default: %(default)s)')
    server_group.add_argument('--timeout', type=int, default=30,
                             help='Request timeout in seconds (default: %(default)s)')
    server_group.add_argument('--health-check', action='store_true',
                             help='Check server health before submission')
    
    # Solver Configuration  
    solver_group = parser.add_argument_group('Solver Configuration')
    solver_group.add_argument('--solver', default='gurobi',
                             help='Solver type (default: %(default)s)')
    solver_group.add_argument('--time-limit', type=int, default=300,
                             help='Max solve time in seconds (default: %(default)s)')
    solver_group.add_argument('--gap', type=float, default=1e-4,
                             help='Optimality gap tolerance (default: %(default)s)')
    solver_group.add_argument('--threads', type=int,
                             help='Number of threads (default: auto)')
    solver_group.add_argument('--verbose-solver', action='store_true',
                             help='Enable solver verbose output')
    
    # Polling Configuration
    poll_group = parser.add_argument_group('Polling Configuration')
    poll_group.add_argument('--max-attempts', type=int, default=60,
                           help='Max polling attempts (default: %(default)s)')
    poll_group.add_argument('--poll-interval', type=float, default=1.0,
                           help='Seconds between polls (default: %(default)s)')
    poll_group.add_argument('--max-wait', type=int, default=600,
                           help='Max total wait time in seconds (default: %(default)s)')
    poll_group.add_argument('--no-poll', action='store_true',
                           help='Submit only, don\'t wait for results')
    
    # Output & Formatting
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='Verbose output')
    output_group.add_argument('--quiet', '-q', action='store_true',
                             help='Minimal output (only final result)')
    output_group.add_argument('--format', choices=['text', 'json', 'csv'], default='text',
                             help='Output format (default: %(default)s)')
    output_group.add_argument('--output', metavar='FILE',
                             help='Write results to file')
    output_group.add_argument('--show-vars', action='store_true',
                             help='Show all variable assignments')
    output_group.add_argument('--show-constraints', action='store_true',
                             help='Show constraint violations (if any)')
    
    # Python Module Options
    python_group = parser.add_argument_group('Python Module Options')
    python_group.add_argument('--python-args', 
                             help='Function arguments as JSON string')
    python_group.add_argument('--python-path', 
                             help='Add path to Python sys.path')
    
    return parser 