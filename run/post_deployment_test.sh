#!/bin/bash

# ROLEX Post-Deployment Test Script
# Comprehensive validation of ROLEX deployment with all solvers

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$SCRIPT_DIR/post_deployment_test.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "$LOG_FILE"
}

# Get server URL from command line or use default
SERVER_URL="${1:-http://localhost:8000}"

# Test functions
test_health_endpoint() {
    log "Testing health endpoint..."
    
    local response=$(curl -s -w "HTTP_CODE:%{http_code}" "$SERVER_URL/health" 2>/dev/null || echo "HTTP_CODE:000")
    local http_code=$(echo "$response" | grep -o "HTTP_CODE:[0-9]*" | cut -d: -f2)
    local body=$(echo "$response" | sed 's/HTTP_CODE:[0-9]*$//')
    
    if [ "$http_code" = "200" ]; then
        if echo "$body" | grep -q "healthy"; then
            log "✅ Health endpoint responding correctly"
            info "Response: $body"
        else
            error "❌ Health endpoint response unexpected: $body"
            return 1
        fi
    else
        error "❌ Health endpoint failed with HTTP code: $http_code"
        return 1
    fi
}

test_solvers_endpoint() {
    log "Testing solvers endpoint..."
    
    local response=$(curl -s -w "HTTP_CODE:%{http_code}" "$SERVER_URL/solvers" 2>/dev/null || echo "HTTP_CODE:000")
    local http_code=$(echo "$response" | grep -o "HTTP_CODE:[0-9]*" | cut -d: -f2)
    local body=$(echo "$response" | sed 's/HTTP_CODE:[0-9]*$//')
    
    if [ "$http_code" = "200" ]; then
        log "✅ Solvers endpoint responding"
        
        # Parse JSON response
        echo "$body" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print('📊 Solver Status:')
    
    for solver, info in data.items():
        status = '✅ Available' if info.get('available', False) else '❌ Not Available'
        version = info.get('version', 'unknown')
        print(f'  • {solver.title()}: {status} (v{version})')
        
        if not info.get('available', False) and 'error' in info:
            print(f'    Error: {info[\"error\"]}')
    
    # Count available solvers
    available = sum(1 for solver in data.values() if solver.get('available', False))
    total = len(data)
    print(f'📈 Summary: {available}/{total} solvers available')
    
except Exception as e:
    print(f'❌ Error parsing solver response: {e}')
    sys.exit(1)
"
    else
        error "❌ Solvers endpoint failed with HTTP code: $http_code"
        return 1
    fi
}

test_cuopt_specifically() {
    log "Testing cuOpt solver specifically..."
    
    local response=$(curl -s "$SERVER_URL/solvers" 2>/dev/null || echo '{}')
    
    local cuopt_available=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    cuopt = data.get('cuopt', {})
    print('true' if cuopt.get('available', False) else 'false')
except:
    print('false')
")
    
    if [ "$cuopt_available" = "true" ]; then
        log "✅ cuOpt solver is available!"
        
        # Get detailed cuOpt info
        echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    cuopt = data.get('cuopt', {})
    print('🔧 cuOpt Details:')
    print(f'  Version: {cuopt.get(\"version\", \"unknown\")}')
    print(f'  GPU Required: {cuopt.get(\"gpu_required\", \"unknown\")}')
    
    if 'gpu_devices' in cuopt:
        print('  GPU Devices:')
        for device in cuopt['gpu_devices']:
            name = device.get('name', 'unknown')
            capability = device.get('compute_capability', 'unknown')
            compatible = device.get('compatible', False)
            status = '✅ Compatible' if compatible else '❌ Not Compatible'
            print(f'    • {name} (Capability: {capability}) - {status}')
    
    if 'features' in cuopt:
        print('  Features:')
        for feature in cuopt['features']:
            print(f'    • {feature}')
            
except Exception as e:
    print(f'❌ Error parsing cuOpt details: {e}')
"
    else
        error "❌ cuOpt solver is NOT available"
        
        # Get error details
        echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    cuopt = data.get('cuopt', {})
    if 'error' in cuopt:
        print(f'❌ cuOpt Error: {cuopt[\"error\"]}')
    else:
        print('❌ cuOpt not available - no specific error message')
except:
    print('❌ Could not parse cuOpt error details')
"
        return 1
    fi
}

test_optimization_request() {
    log "Testing optimization request..."
    
    # Create a simple optimization problem
    local problem_data='{
        "variables": [
            {"id": 1, "name": "x1", "type": "continuous", "lower": 0, "upper": 10},
            {"id": 2, "name": "x2", "type": "continuous", "lower": 0, "upper": 10}
        ],
        "constraints": [
            {"id": 1, "name": "constraint1", "terms": [{"variable": 1, "coefficient": 1.0}, {"variable": 2, "coefficient": 1.0}], "upper": 1.0, "equality": false}
        ],
        "objective": {
            "id": 1, 
            "name": "objective",
            "terms": [{"variable": 1, "coefficient": 1.0}, {"variable": 2, "coefficient": 1.0}],
            "sense": "maximize"
        }
    }'
    
    info "Sending optimization request..."
    
    local response=$(curl -s -w "HTTP_CODE:%{http_code}" \
        -H "Content-Type: application/json" \
        -d "$problem_data" \
        "$SERVER_URL/optimize" 2>/dev/null || echo "HTTP_CODE:000")
    
    local http_code=$(echo "$response" | grep -o "HTTP_CODE:[0-9]*" | cut -d: -f2)
    local body=$(echo "$response" | sed 's/HTTP_CODE:[0-9]*$//')
    
    if [ "$http_code" = "200" ]; then
        log "✅ Optimization request successful"
        
        # Parse optimization result
        echo "$body" | python3 -c "
import sys, json
try:
    result = json.load(sys.stdin)
    print('📊 Optimization Result:')
    print(f'  Status: {result.get(\"status\", \"unknown\")}')
    print(f'  Objective Value: {result.get(\"objective_value\", \"unknown\")}')
    print(f'  Solver: {result.get(\"solver\", \"unknown\")}')
    print(f'  Solve Time: {result.get(\"solve_time\", \"unknown\")}s')
    
    if 'variables' in result:
        print('  Variables:')
        for var, value in result['variables'].items():
            print(f'    {var} = {value}')
    
    if result.get('status') == 'optimal':
        print('✅ Optimization completed successfully!')
    else:
        print('❌ Optimization did not reach optimal solution')
        
except Exception as e:
    print(f'❌ Error parsing optimization result: {e}')
    sys.exit(1)
"
    else
        error "❌ Optimization request failed with HTTP code: $http_code"
        error "Response: $body"
        return 1
    fi
}

test_docs_endpoint() {
    log "Testing API documentation endpoint..."
    
    local response=$(curl -s -w "HTTP_CODE:%{http_code}" "$SERVER_URL/docs" 2>/dev/null || echo "HTTP_CODE:000")
    local http_code=$(echo "$response" | grep -o "HTTP_CODE:[0-9]*" | cut -d: -f2)
    
    if [ "$http_code" = "200" ]; then
        log "✅ API documentation accessible"
        info "Docs URL: $SERVER_URL/docs"
    else
        warning "⚠️ API documentation not accessible (HTTP code: $http_code)"
    fi
}

test_openapi_spec() {
    log "Testing OpenAPI specification..."
    
    local response=$(curl -s -w "HTTP_CODE:%{http_code}" "$SERVER_URL/openapi.json" 2>/dev/null || echo "HTTP_CODE:000")
    local http_code=$(echo "$response" | grep -o "HTTP_CODE:[0-9]*" | cut -d: -f2)
    local body=$(echo "$response" | sed 's/HTTP_CODE:[0-9]*$//')
    
    if [ "$http_code" = "200" ]; then
        log "✅ OpenAPI specification accessible"
        
        # Parse OpenAPI spec
        echo "$body" | python3 -c "
import sys, json
try:
    spec = json.load(sys.stdin)
    print('📋 API Specification:')
    print(f'  Title: {spec.get(\"info\", {}).get(\"title\", \"unknown\")}')
    print(f'  Version: {spec.get(\"info\", {}).get(\"version\", \"unknown\")}')
    
    if 'paths' in spec:
        print(f'  Endpoints: {len(spec[\"paths\"])}')
        for path in spec['paths']:
            print(f'    • {path}')
            
except Exception as e:
    print(f'❌ Error parsing OpenAPI spec: {e}')
"
    else
        warning "⚠️ OpenAPI specification not accessible (HTTP code: $http_code)"
    fi
}

run_server_verification() {
    log "Running server verification on the deployed instance..."
    
    # Check if verify_solvers.py exists
    if [ -f "$PROJECT_ROOT/server/verify_solvers.py" ]; then
        log "Running comprehensive solver verification..."
        
        # Try to run the verification script remotely
        # This would require SSH access to the server
        warning "Local solver verification script found but requires server access"
        info "To run on server: ssh ubuntu@<server-ip> 'cd /home/ubuntu/rolex && python3 server/verify_solvers.py'"
    else
        warning "Solver verification script not found"
    fi
}

# Main test execution
main() {
    # Create log file
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "ROLEX Post-Deployment Test Log - $(date)" > "$LOG_FILE"
    
    log "🧪 Starting ROLEX post-deployment tests..."
    log "Server URL: $SERVER_URL"
    echo "=" >> "$LOG_FILE"
    
    # Track test results
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    
    # Run tests
    tests=(
        "test_health_endpoint"
        "test_solvers_endpoint"
        "test_cuopt_specifically"
        "test_optimization_request"
        "test_docs_endpoint"
        "test_openapi_spec"
    )
    
    for test in "${tests[@]}"; do
        total_tests=$((total_tests + 1))
        log "Running $test..."
        
        if $test; then
            passed_tests=$((passed_tests + 1))
            log "✅ $test PASSED"
        else
            failed_tests=$((failed_tests + 1))
            error "❌ $test FAILED"
        fi
        
        echo "-" >> "$LOG_FILE"
    done
    
    # Additional verification
    run_server_verification
    
    # Summary
    log "📊 Test Summary"
    echo "=" >> "$LOG_FILE"
    log "Total Tests: $total_tests"
    log "Passed: $passed_tests"
    log "Failed: $failed_tests"
    log "Success Rate: $(( passed_tests * 100 / total_tests ))%"
    
    if [ $failed_tests -eq 0 ]; then
        log "🎉 All tests passed! ROLEX deployment is working correctly."
        log "✅ Server is ready for production use."
        exit 0
    else
        error "❌ $failed_tests test(s) failed. Please check the issues above."
        exit 1
    fi
}

# Handle script interruption
trap 'error "Tests interrupted"; exit 1' INT TERM

# Show usage if no arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 [server_url]"
    echo "Example: $0 http://52.59.152.54:8000"
    echo "Default: http://localhost:8000"
    echo
fi

# Run main function
main "$@" 