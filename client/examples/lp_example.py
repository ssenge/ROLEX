#!/usr/bin/env python3
"""
Example using ROLEX with LP file format.
"""

import sys
sys.path.insert(0, '..')
import client

def main():
    """Main example function demonstrating LP file conversion."""
    print("🚀 ROLEX LP File Example")
    print("=" * 50)
    
    # Method 1: Direct conversion using Converter
    print("\n📁 Method 1: Direct Converter Usage")
    try:
        instance = client.Converter.from_lp("../test_files/example.lp")
        problem = client.Problem.from_instance(instance)
        print(f"✅ Converted LP file to problem: {problem}")
        print(f"🔗 Variable mapping: {problem.get_variable_mapping()}")
    except Exception as e:
        print(f"❌ Error with direct conversion: {e}")
    
    # Method 2: Using Problem.from_file convenience method
    print("\n📁 Method 2: Problem.from_file() Convenience Method")
    try:
        problem = client.Problem.from_file("../test_files/example.lp")
        print(f"✅ Loaded LP file directly: {problem}")
        print(f"🔗 Variable mapping: {problem.get_variable_mapping()}")
    except Exception as e:
        print(f"❌ Error with convenience method: {e}")
    
    # Method 3: Auto-detection
    print("\n📁 Method 3: Auto-detection")
    try:
        problem = client.Problem.from_file("../test_files/example.lp")
        print(f"✅ Auto-detected LP format: {problem}")
        
        # Create client and submit
        rolex_client = client.Client()
        
        if rolex_client.health_check():
            print("✅ Server is healthy - submitting problem...")
            
            job_id = rolex_client.submit(problem)
            print(f"✅ Job submitted: {job_id}")
            
            result = rolex_client.poll(job_id, problem)
            print(f"📊 Result: {result}")
            
            if result.is_optimal():
                print(f"\n🎯 Solution Details:")
                print(f"  Variables: {result.get_variables()}")
                print(f"  Objective: {result.objective_value}")
            
        else:
            print("❌ Server not available - skipping solve")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 