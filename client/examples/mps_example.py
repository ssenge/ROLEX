#!/usr/bin/env python3
"""
Example using ROLEX with MPS file format.
"""

import sys
sys.path.insert(0, '..')
import client

def main():
    """Main example function demonstrating MPS file conversion."""
    print("🚀 ROLEX MPS File Example")
    print("=" * 50)
    
    # Method 1: Direct conversion using Converter
    print("\n📁 Method 1: Direct Converter Usage")
    try:
        instance = client.Converter.from_mps("../test_files/example.mps")
        problem = client.Problem.from_instance(instance)
        print(f"✅ Converted MPS file to problem: {problem}")
        print(f"🔗 Variable mapping: {problem.get_variable_mapping()}")
    except Exception as e:
        print(f"❌ Error with direct conversion: {e}")
    
    # Method 2: Using Problem.from_file convenience method
    print("\n📁 Method 2: Problem.from_file() Convenience Method")
    try:
        problem = client.Problem.from_file("../test_files/example.mps")
        print(f"✅ Loaded MPS file directly: {problem}")
        print(f"🔗 Variable mapping: {problem.get_variable_mapping()}")
    except Exception as e:
        print(f"❌ Error with convenience method: {e}")
    
    # Method 3: Format comparison
    print("\n📁 Method 3: Format Comparison")
    try:
        lp_problem = rolex.Problem.from_file("../test_files/example.lp")
        mps_problem = rolex.Problem.from_file("../test_files/example.mps")
        
        print(f"LP Problem: {lp_problem}")
        print(f"MPS Problem: {mps_problem}")
        
        print(f"LP Variables: {lp_problem.get_variable_mapping()}")
        print(f"MPS Variables: {mps_problem.get_variable_mapping()}")
        
        # Test format detection
        lp_format = client.Converter._detect_format("../test_files/example.lp")
        mps_format = client.Converter._detect_format("../test_files/example.mps")
        
        print(f"Detected LP format: {lp_format}")
        print(f"Detected MPS format: {mps_format}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 