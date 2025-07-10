#!/usr/bin/env python3
"""
Comprehensive example demonstrating all ROLEX Converter functionality.
"""

import sys
import os
sys.path.insert(0, '..')
import client

def test_converter_functionality():
    """Test all converter methods and error handling."""
    print("🚀 ROLEX Converter Test Suite")
    print("=" * 50)
    
    # Test 1: File format detection
    print("\n📂 Test 1: File Format Detection")
    test_files = [
        "../test_files/example.lp",
        "../test_files/example.mps",
        "nonexistent.lp",
        "../test_files/example.lp"  # Test same file twice
    ]
    
    for file_path in test_files:
        try:
            if os.path.exists(file_path):
                format_detected = client.Converter._detect_format(file_path)
                print(f"  ✅ {file_path} -> {format_detected}")
            else:
                print(f"  ❌ {file_path} -> File not found")
        except Exception as e:
            print(f"  ❌ {file_path} -> Error: {e}")
    
    # Test 2: LP file conversion
    print("\n📄 Test 2: LP File Conversion")
    try:
        instance = client.Converter.from_lp("../test_files/example.lp")
        problem = client.Problem.from_instance(instance)
        print(f"  ✅ LP conversion successful: {problem}")
        print(f"  🔗 Variables: {problem.get_variable_mapping()}")
    except Exception as e:
        print(f"  ❌ LP conversion failed: {e}")
    
    # Test 3: MPS file conversion
    print("\n📄 Test 3: MPS File Conversion")
    try:
        instance = client.Converter.from_mps("../test_files/example.mps")
        problem = client.Problem.from_instance(instance)
        print(f"  ✅ MPS conversion successful: {problem}")
        print(f"  🔗 Variables: {problem.get_variable_mapping()}")
    except Exception as e:
        print(f"  ❌ MPS conversion failed: {e}")
    
    # Test 4: Auto-detection
    print("\n🔍 Test 4: Auto-detection")
    auto_test_files = ["../test_files/example.lp", "../test_files/example.mps"]
    
    for file_path in auto_test_files:
        try:
            problem = client.Problem.from_file(file_path)
            print(f"  ✅ Auto-detected {file_path}: {problem}")
        except Exception as e:
            print(f"  ❌ Auto-detection failed for {file_path}: {e}")
    
    # Test 5: Error handling
    print("\n⚠️  Test 5: Error Handling")
    error_tests = [
        ("nonexistent.lp", "File not found"),
        ("../test_files/", "Not a file"),
    ]
    
    for file_path, expected_error in error_tests:
        try:
            client.Converter.from_file(file_path)
            print(f"  ❌ Expected error for {file_path}, but got success")
        except client.FileNotFoundError as e:
            print(f"  ✅ Expected FileNotFoundError for {file_path}: {e}")
        except client.FileFormatError as e:
            print(f"  ✅ Expected FileFormatError for {file_path}: {e}")
        except Exception as e:
            print(f"  ⚠️  Unexpected error for {file_path}: {e}")
    
    # Test 6: Integration with Client
    print("\n🔄 Test 6: Integration with Client")
    try:
        problem = client.Problem.from_file("../test_files/example.lp")
        rolex_client = client.Client()
        
        if rolex_client.health_check():
            print("  ✅ Server is healthy - testing full workflow...")
            
            job_id = rolex_client.submit(problem)
            print(f"  ✅ Job submitted: {job_id}")
            
            result = rolex_client.poll(job_id, problem)
            print(f"  📊 Result: {result}")
            
            if result.is_optimal():
                print(f"  🎯 Solution: {result.get_variables()}")
                print(f"  🔢 Objective: {result.objective_value}")
                
        else:
            print("  ❌ Server not available - skipping integration test")
            
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")

def main():
    """Run all converter tests."""
    test_converter_functionality()
    
    print("\n" + "=" * 50)
    print("🎉 Converter Test Suite Complete!")
    print("\n💡 Usage Examples:")
    print("  # Direct conversion:")
    print("  instance = client.Converter.from_lp('file.lp')")
    print("  problem = client.Problem.from_instance(instance)")
    print("")
    print("  # Convenience method:")
    print("  problem = client.Problem.from_file('file.lp')")
    print("")
    print("  # Auto-detection:")
    print("  problem = client.Problem.from_file('unknown_format.txt')")

if __name__ == "__main__":
    main() 