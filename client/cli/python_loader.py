"""
Python module loading for ROLEX CLI.
"""

import importlib
import importlib.util
import sys
import os
import json
import re
from typing import Any
import ommx.v1 as ommx

def load_python_instance(args) -> ommx.Instance:
    """Load OMMX instance from Python module."""
    
    # Parse module:function syntax
    module_path, func_name, func_args_str = parse_python_spec(args.python)
    
    # Add custom paths
    if args.python_path:
        sys.path.insert(0, args.python_path)
    
    # Import module
    module = import_module(module_path)
    
    # Get function
    func = getattr(module, func_name)
    
    # Parse arguments
    func_args = parse_function_args(args.python_args, func_args_str)
    
    # Call function
    if func_args:
        instance = func(**func_args)
    else:
        instance = func()
    
    # Validate return type
    if not isinstance(instance, ommx.Instance):
        raise ValueError(f"Function must return ommx.Instance, got {type(instance)}")
    
    return instance

def parse_python_spec(python_spec: str):
    """Parse Python module specification."""
    # Handle formats:
    # - "module.py:function"
    # - "module:function"  
    # - "module.py:function(args)"
    # - "module:function(args)"
    
    # Check for function arguments in parentheses
    func_args_match = re.search(r'\(([^)]*)\)$', python_spec)
    func_args_str = func_args_match.group(1) if func_args_match else None
    
    # Remove function arguments for parsing
    if func_args_match:
        python_spec = python_spec[:func_args_match.start()]
    
    # Split module and function
    if ':' not in python_spec:
        raise ValueError("Python spec must be in format 'module:function'")
    
    module_path, func_name = python_spec.split(':', 1)
    
    return module_path, func_name, func_args_str

def import_module(module_path: str):
    """Import module from path."""
    if module_path.endswith('.py'):
        # Import from file
        module_name = os.path.basename(module_path)[:-3]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            raise ImportError(f"Cannot load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        # Import from module name
        module = importlib.import_module(module_path)
    
    return module

def parse_function_args(json_args: str = None, inline_args: str = None):
    """Parse function arguments from JSON or inline format."""
    if json_args:
        return json.loads(json_args)
    elif inline_args:
        # Simple parsing for inline arguments like "cities=50, seed=42"
        args = {}
        if inline_args.strip():
            for arg in inline_args.split(','):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Try to parse as number
                    try:
                        if '.' in value:
                            args[key] = float(value)
                        else:
                            args[key] = int(value)
                    except ValueError:
                        # Keep as string, removing quotes
                        args[key] = value.strip('"\'')
        return args
    return {} 