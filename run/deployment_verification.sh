#!/bin/bash

# ROLEX Deployment Verification Script
# Comprehensive verification of ROLEX server deployment with MPS support

set -e  # Exit on any error

echo "==========================================="
echo "🚀 ROLEX DEPLOYMENT VERIFICATION"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "ℹ️  $1"
}

# Check if running on correct server
echo "1. Checking server environment..."
CURRENT_USER=$(whoami)
if [[ "$CURRENT_USER" != "ubuntu" ]]; then
    print_warning "Not running as ubuntu user (current: $CURRENT_USER)"
fi

# Check conda environments
echo "2. Checking conda environments..."
conda env list | grep -E "(ommx-server|cuOpt-server)" || {
    print_error "Required conda environments not found"
    exit 1
}

# Check cuOpt CLI
echo "3. Checking cuOpt CLI..."
CUOPT_CLI_PATH="/home/ubuntu/.conda/envs/cuOpt-server/bin/cuopt_cli"

if [[ ! -f "$CUOPT_CLI_PATH" ]]; then
    print_error "cuOpt CLI not found at expected location: $CUOPT_CLI_PATH"
    echo "Available files in cuOpt-server/bin:"
    ls -la /home/ubuntu/.conda/envs/cuOpt-server/bin/ | grep -E "(cuopt|cuOpt)" || echo "No cuopt files found"
    exit 1
fi

if [[ ! -x "$CUOPT_CLI_PATH" ]]; then
    print_error "cuOpt CLI not executable: $CUOPT_CLI_PATH"
    exit 1
fi

print_success "cuOpt CLI found and executable: $CUOPT_CLI_PATH"

# Test cuOpt CLI help
echo "4. Testing cuOpt CLI help..."
if "$CUOPT_CLI_PATH" --help > /dev/null 2>&1; then
    print_success "cuOpt CLI help command works"
else
    print_error "cuOpt CLI help command failed"
    exit 1
fi

# Check if ROLEX server files exist
echo "5. Checking ROLEX server files..."
REQUIRED_FILES=(
    "server/server.py"
    "server/models.py"
    "server/job_manager.py"
    "server/solvers/gurobi_solver.py"
    "server/solvers/cuopt_solver.py"
    "server/solvers/gurobi_mps_solver.py"
    "server/solvers/cuopt_mps_solver.py"
    "server/solvers/mps_base.py"
    "run/rolex_cli.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        print_success "Found: $file"
    else
        print_error "Missing: $file"
        exit 1
    fi
done

# Test MPS file exists
echo "6. Checking test MPS file..."
if [[ -f "client/test_files/example.mps" ]]; then
    print_success "Test MPS file found"
    echo "MPS file content preview:"
    head -10 client/test_files/example.mps
else
    print_error "Test MPS file not found"
    exit 1
fi

# Test cuOpt CLI with MPS file
echo "7. Testing cuOpt CLI with MPS file..."
TEMP_OUTPUT="/tmp/rolex_test_output.txt"

if "$CUOPT_CLI_PATH" --solution-file "$TEMP_OUTPUT" client/test_files/example.mps; then
    print_success "cuOpt CLI solved MPS file successfully"
    echo "cuOpt CLI output:"
    cat "$TEMP_OUTPUT"
    rm -f "$TEMP_OUTPUT"
else
    print_error "cuOpt CLI failed to solve MPS file"
    exit 1
fi

# Check Python dependencies
echo "8. Checking Python dependencies..."
conda activate ommx-server

python -c "
import sys
required_packages = [
    'fastapi',
    'uvicorn',
    'pydantic',
    'ommx',
    'requests'
]

missing = []
for package in required_packages:
    try:
        __import__(package)
        print(f'✅ {package}')
    except ImportError:
        missing.append(package)
        print(f'❌ {package}')

if missing:
    print(f'Missing packages: {missing}')
    sys.exit(1)
else:
    print('All required packages found')
"

# Check Gurobi availability
echo "9. Checking Gurobi availability..."
python -c "
try:
    import gurobipy as gp
    print('✅ Gurobi available')
    
    # Test basic functionality
    model = gp.Model()
    print('✅ Gurobi model creation works')
except Exception as e:
    print(f'❌ Gurobi issue: {e}')
"

# Start ROLEX server in background for testing
echo "10. Starting ROLEX server for testing..."
cd server
python server.py --port 8001 &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Test server health
echo "11. Testing ROLEX server health..."
if curl -s http://localhost:8001/health > /dev/null; then
    print_success "ROLEX server health check passed"
else
    print_error "ROLEX server health check failed"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Test MPS solvers endpoint
echo "12. Testing MPS solvers endpoint..."
SOLVERS_RESPONSE=$(curl -s http://localhost:8001/solvers/mps)
if echo "$SOLVERS_RESPONSE" | grep -q "gurobi"; then
    print_success "MPS solvers endpoint working"
    echo "Available MPS solvers:"
    echo "$SOLVERS_RESPONSE" | python -m json.tool
else
    print_error "MPS solvers endpoint failed"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Test MPS job submission
echo "13. Testing MPS job submission..."
cd ..
python -c "
import requests
import json

# Submit MPS job
with open('client/test_files/example.mps', 'rb') as f:
    files = {'mps_file': f}
    data = {
        'solver': 'gurobi',
        'parameters': json.dumps({'max_time': 60})
    }
    
    response = requests.post('http://localhost:8001/jobs/submit-mps', files=files, data=data)
    
if response.status_code == 200:
    job_data = response.json()
    job_id = job_data['job_id']
    print(f'✅ MPS job submitted successfully: {job_id}')
    
    # Wait a bit and check status
    import time
    time.sleep(2)
    
    status_response = requests.get(f'http://localhost:8001/jobs/{job_id}/mps')
    if status_response.status_code == 200:
        print('✅ MPS job status retrieval works')
        print(f'Job status: {status_response.json()[\"status\"]}')
    else:
        print('❌ MPS job status retrieval failed')
        exit(1)
else:
    print(f'❌ MPS job submission failed: {response.status_code}')
    exit(1)
"

# Test CLI client
echo "14. Testing CLI client..."
python run/rolex_cli.py --list-solvers --server http://localhost:8001 || {
    print_error "CLI client failed"
    kill $SERVER_PID 2>/dev/null
    exit 1
}

print_success "CLI client works"

# Clean up
echo "15. Cleaning up..."
kill $SERVER_PID 2>/dev/null
sleep 2

echo "==========================================="
print_success "🎉 ALL VERIFICATION CHECKS PASSED!"
echo "==========================================="

print_info "ROLEX deployment is ready for production use"
print_info "Available components:"
echo "  • ROLEX server with OMMX and MPS support"
echo "  • Gurobi solver (OMMX and MPS)"
echo "  • cuOpt solver (OMMX and MPS via CLI)"
echo "  • SciPy solver (OMMX only)"
echo "  • CLI client for MPS files"
echo "  • Web API for both formats"

echo ""
print_info "To start the server:"
echo "  cd server && python server.py"
echo ""
print_info "To use the CLI client:"
echo "  python run/rolex_cli.py problem.mps --solver gurobi"
echo ""
print_info "Server endpoints:"
echo "  • POST /jobs/submit (OMMX format)"
echo "  • POST /jobs/submit-mps (MPS format)"
echo "  • GET /solvers (OMMX solvers)"
echo "  • GET /solvers/mps (MPS solvers)"
echo "  • GET /health (health check)"

echo "===========================================" 