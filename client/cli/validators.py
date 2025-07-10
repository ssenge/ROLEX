"""
Input validation for ROLEX CLI.
"""

import os
import sys

def validate_inputs(args):
    """Validate command-line arguments."""
    
    # Validate input source
    if args.input_file:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        if not os.path.isfile(args.input_file):
            raise ValueError(f"Input path is not a file: {args.input_file}")
    
    # Validate Python specification
    if args.python:
        if ':' not in args.python:
            raise ValueError("Python spec must be in format 'module:function'")
    
    # Validate conflicting options
    if args.quiet and args.verbose:
        raise ValueError("Cannot use both --quiet and --verbose")
    
    # Validate polling options
    if args.no_poll and (args.max_attempts != 60 or args.poll_interval != 1.0):
        print("Warning: Polling options ignored when --no-poll is used")
    
    # Validate output format
    if args.format != 'text' and args.quiet:
        raise ValueError("Cannot use --quiet with non-text output formats")
    
    # Validate file output
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if not os.path.exists(output_dir):
            raise ValueError(f"Output directory does not exist: {output_dir}")
    
    # Validate Python path
    if args.python_path and not os.path.exists(args.python_path):
        raise ValueError(f"Python path does not exist: {args.python_path}")
    
    # Validate numerical values
    if args.time_limit <= 0:
        raise ValueError("Time limit must be positive")
    
    if args.gap < 0:
        raise ValueError("Gap must be non-negative")
    
    if args.timeout <= 0:
        raise ValueError("Timeout must be positive")
    
    if args.max_attempts <= 0:
        raise ValueError("Max attempts must be positive")
    
    if args.poll_interval <= 0:
        raise ValueError("Poll interval must be positive")
    
    if args.max_wait <= 0:
        raise ValueError("Max wait time must be positive")
    
    if args.threads is not None and args.threads <= 0:
        raise ValueError("Number of threads must be positive") 