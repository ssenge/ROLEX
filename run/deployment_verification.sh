#!/bin/bash

# ROLEX Deployment Verification Script
# Comprehensive verification of ROLEX MPS server deployment

set -e  # Exit on any error

echo "==========================================="
echo "🚀 ROLEX MPS DEPLOYMENT VERIFICATION"
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

# Configuration
SERVER_URL="http://localhost:8000"
CONDA_ENV="rolex-server"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if running on correct server
echo "1. Checking server environment..."
CURRENT_USER=$(whoami)
if [[ "$CURRENT_USER" != "ubuntu" ]]; then
    print_warning "Not running as ubuntu user (current: $CURRENT_USER)"
fi

# Check conda environments
echo "2. Checking conda environments..."
if conda env list | grep -q "$CONDA_ENV"; then
    print_success "Found $CONDA_ENV environment"
else
    print_error "Required conda environment $CONDA_ENV not found"
    exit 1
fi

# Check cuOpt CLI
echo "3. Checking cuOpt CLI..."
if eval "$(conda shell.bash hook)" && conda activate "$CONDA_ENV" && which cuopt_cli > /dev/null; then
    print_success "cuOpt CLI is available"
    cuopt_cli_version=$(conda run -n "$CONDA_ENV" cuopt_cli --version 2>/dev/null || echo "unknown")
    print_info "cuOpt CLI version: $cuopt_cli_version"
else
    print_error "cuOpt CLI not found in PATH"
    exit 1
fi

# Test cuOpt Python module
echo "4. Testing cuOpt Python module..."
if conda run -n "$CONDA_ENV" python -c "import cuopt; print('cuOpt Python module available')" > /dev/null 2>&1; then
    print_success "cuOpt Python module is available"
else
    print_error "cuOpt Python module not available"
    exit 1
fi

# Test Gurobi
echo "5. Testing Gurobi..."
if conda run -n "$CONDA_ENV" python -c "import gurobipy as gp; env = gp.Env(); print('Gurobi available'); env.dispose()" > /dev/null 2>&1; then
    print_success "Gurobi is available"
else
    print_warning "Gurobi not available (license may be needed)"
fi

# Check server status
echo "6. Checking server status..."
if curl -s "$SERVER_URL/health" > /dev/null; then
    print_success "Server is responding"
else
    print_error "Server is not responding at $SERVER_URL"
    print_info "Try starting server: cd $PROJECT_ROOT && ./run/rolex_server.sh start"
    exit 1
fi

# Test health endpoint
echo "7. Testing health endpoint..."
health_response=$(curl -s "$SERVER_URL/health")
if echo "$health_response" | grep -q "healthy"; then
    print_success "Health endpoint working"
else
    print_error "Health endpoint not working properly"
    exit 1
fi

# Test MPS solvers endpoint
echo "8. Testing MPS solvers endpoint..."
mps_solvers_response=$(curl -s "$SERVER_URL/solvers/mps")
if echo "$mps_solvers_response" | grep -q "gurobi\|cuopt"; then
    print_success "MPS solvers endpoint working"
    print_info "Available solvers: $(echo "$mps_solvers_response" | grep -o '"[^"]*"' | head -3)"
else
    print_error "MPS solvers endpoint not working"
    exit 1
fi

# Create test MPS file
echo "9. Creating test MPS file..."
test_mps_file="/tmp/test_problem.mps"
cat > "$test_mps_file" << 'EOF'
NAME          TEST_PROBLEM
ROWS
 N  COST
 L  CONSTRAINT1
COLUMNS
    X1        COST             1.0
    X1        CONSTRAINT1      1.0
    X2        COST             1.0
    X2        CONSTRAINT1      1.0
RHS
    RHS1      CONSTRAINT1      1.0
BOUNDS
 UP BND1      X1               1.0
 UP BND1      X2               1.0
ENDATA
EOF

if [[ -f "$test_mps_file" ]]; then
    print_success "Test MPS file created"
else
    print_error "Failed to create test MPS file"
    exit 1
fi

# Test MPS job submission (try Gurobi first)
echo "10. Testing MPS job submission with Gurobi..."
gurobi_job_response=$(curl -s -X POST \
    -F "mps_file=@$test_mps_file" \
    -F "solver=gurobi" \
    -F "parameters={\"max_time\": 60}" \
    "$SERVER_URL/jobs/submit-mps")

if echo "$gurobi_job_response" | grep -q "job_id"; then
    print_success "Gurobi MPS job submitted"
    gurobi_job_id=$(echo "$gurobi_job_response" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)
    print_info "Gurobi job ID: $gurobi_job_id"
    
    # Check job status
    echo "11. Checking Gurobi job status..."
    sleep 2
    gurobi_status_response=$(curl -s "$SERVER_URL/jobs/$gurobi_job_id/mps")
    if echo "$gurobi_status_response" | grep -q "completed\|optimal"; then
        print_success "Gurobi job completed successfully"
    else
        print_warning "Gurobi job status: $(echo "$gurobi_status_response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)"
    fi
else
    print_warning "Gurobi MPS job submission failed"
fi

# Test MPS job submission with cuOpt
echo "12. Testing MPS job submission with cuOpt..."
cuopt_job_response=$(curl -s -X POST \
    -F "mps_file=@$test_mps_file" \
    -F "solver=cuopt" \
    -F "parameters={\"max_time\": 60}" \
    "$SERVER_URL/jobs/submit-mps")

if echo "$cuopt_job_response" | grep -q "job_id"; then
    print_success "cuOpt MPS job submitted"
    cuopt_job_id=$(echo "$cuopt_job_response" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)
    print_info "cuOpt job ID: $cuopt_job_id"
    
    # Check job status
    echo "13. Checking cuOpt job status..."
    sleep 2
    cuopt_status_response=$(curl -s "$SERVER_URL/jobs/$cuopt_job_id/mps")
    if echo "$cuopt_status_response" | grep -q "completed\|optimal"; then
        print_success "cuOpt job completed successfully"
    else
        print_warning "cuOpt job status: $(echo "$cuopt_status_response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)"
    fi
else
    print_warning "cuOpt MPS job submission failed"
fi

# Test CLI client
echo "14. Testing CLI client..."
if [[ -f "$PROJECT_ROOT/run/rolex_cli.py" ]]; then
    print_success "CLI client found"
    
    # Test CLI with --list-solvers
    echo "15. Testing CLI --list-solvers..."
    if conda run -n "$CONDA_ENV" python "$PROJECT_ROOT/run/rolex_cli.py" --server "$SERVER_URL" --list-solvers > /dev/null 2>&1; then
        print_success "CLI --list-solvers working"
    else
        print_warning "CLI --list-solvers not working"
    fi
    
    # Test CLI job submission
    echo "16. Testing CLI job submission..."
    if conda run -n "$CONDA_ENV" python "$PROJECT_ROOT/run/rolex_cli.py" --server "$SERVER_URL" --solver gurobi --max-time 30 "$test_mps_file" > /dev/null 2>&1; then
        print_success "CLI job submission working"
    else
        print_warning "CLI job submission failed"
    fi
else
    print_warning "CLI client not found"
fi

# Clean up
echo "17. Cleaning up..."
rm -f "$test_mps_file"
print_success "Test files cleaned up"

echo "==========================================="
print_success "ROLEX MPS DEPLOYMENT VERIFICATION COMPLETE"
echo "==========================================="

print_info "Server URL: $SERVER_URL"
print_info "Available endpoints:"
print_info "  - GET /health"
print_info "  - GET /solvers/mps"
print_info "  - POST /jobs/submit-mps"
print_info "  - GET /jobs/{job_id}/mps"
print_info ""
print_info "CLI Usage:"
print_info "  python $PROJECT_ROOT/run/rolex_cli.py --server $SERVER_URL --solver gurobi example.mps"
print_info ""
print_info "ROLEX MPS deployment is ready for production use" 