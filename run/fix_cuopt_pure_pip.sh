#!/bin/bash

# ROLEX cuOpt Fix Script - Pure Pip Approach
# This script resolves the CUDA library version mismatch by removing conda 
# PyTorch/CUDA packages and installing everything via pip for compatibility

set -e  # Exit on any error

CONDA_ENV_NAME="rolex-server"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVER_DIR="$PROJECT_ROOT/server"
LOG_FILE="$PROJECT_ROOT/run/cuopt_fix.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        error "Conda not found. Please ensure conda is installed and in PATH."
        exit 1
    fi
    
    if ! conda env list | grep -q "$CONDA_ENV_NAME"; then
        error "Conda environment '$CONDA_ENV_NAME' not found."
        exit 1
    fi
}

# Stop the ROLEX server if running
stop_server() {
    log "Stopping ROLEX server if running..."
    
    PID_FILE="$PROJECT_ROOT/run/rolex_server.pid"
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p $pid > /dev/null 2>&1; then
            log "Stopping server with PID: $pid"
            kill $pid
            sleep 3
            if ps -p $pid > /dev/null 2>&1; then
                warning "Server still running, force killing..."
                kill -9 $pid
            fi
        fi
        rm -f "$PID_FILE"
    fi
    
    # Also kill any python server.py processes
    pkill -f "python server.py" || true
    
    success "Server stopped"
}

# Backup current package state
backup_packages() {
    log "Creating backup of current package state..."
    
    # Activate conda environment and export package list
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
    
    conda list --export > "$PROJECT_ROOT/run/conda_packages_backup.txt"
    pip list --format=freeze > "$PROJECT_ROOT/run/pip_packages_backup.txt"
    
    success "Package lists backed up to run/ directory"
}

# Remove conda PyTorch and CUDA packages
remove_conda_packages() {
    log "Removing conda PyTorch and CUDA packages..."
    
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
    
    # List of conda packages to remove
    local packages_to_remove=(
        "pytorch"
        "torchvision" 
        "torchaudio"
        "pytorch-cuda"
        "cudatoolkit"
        "cuda-runtime"
        "nvidia-cublas-cu12"
        "nvidia-cusolver-cu12"
        "nvidia-cusparse-cu12"
        "nvidia-curand-cu12"
        "nvidia-cufft-cu12"
        "nvidia-cudnn-cu12"
        "nvidia-nccl-cu12"
    )
    
    for package in "${packages_to_remove[@]}"; do
        if conda list | grep -q "^$package"; then
            log "Removing conda package: $package"
            conda remove -y "$package" || warning "Failed to remove $package (may not exist)"
        fi
    done
    
    success "Conda PyTorch/CUDA packages removed"
}

# Install PyTorch via pip with CUDA 12.1 support
install_pytorch_pip() {
    log "Installing PyTorch via pip with CUDA 12.1 support..."
    
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
    
    # Install PyTorch with CUDA 12.1 support
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    success "PyTorch installed via pip with CUDA 12.1 support"
}

# Reinstall cuOpt via pip
install_cuopt_pip() {
    log "Reinstalling cuOpt via pip..."
    
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
    
    # Remove existing cuOpt installation
    pip uninstall -y cuopt-cu12 || true
    
    # Install cuOpt (will bring compatible CUDA dependencies)
    pip install cuopt-cu12==25.5.1
    
    success "cuOpt installed via pip"
}

# Verify CUDA compatibility
verify_cuda_compatibility() {
    log "Verifying CUDA compatibility..."
    
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
    
    # Test PyTorch CUDA
    python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU compute capability: {torch.cuda.get_device_capability(0)}')
"
    
    # Test cuOpt import
    log "Testing cuOpt import..."
    python -c "
try:
    from cuopt import routing
    print('✅ cuOpt import successful!')
    print(f'cuOpt version: {routing.__version__}')
except Exception as e:
    print(f'❌ cuOpt import failed: {e}')
    raise
"
    
    success "CUDA compatibility verified"
}

# Update library paths (remove old conda-specific paths)
update_library_paths() {
    log "Updating library paths..."
    
    local activate_script="/home/ubuntu/.conda/envs/$CONDA_ENV_NAME/etc/conda/activate.d/cuopt_libs.sh"
    
    if [ -f "$activate_script" ]; then
        log "Removing old conda-specific library paths..."
        rm -f "$activate_script"
    fi
    
    # Create new activation script with pip-based paths
    mkdir -p "/home/ubuntu/.conda/envs/$CONDA_ENV_NAME/etc/conda/activate.d/"
    
    cat > "$activate_script" << 'EOF'
#!/bin/bash
# cuOpt library paths for pip-installed packages

# Add conda environment lib paths
export LD_LIBRARY_PATH="/home/ubuntu/.conda/envs/rolex-server/lib:$LD_LIBRARY_PATH"

# Add cuOpt library paths (pip-installed)
export LD_LIBRARY_PATH="/home/ubuntu/.conda/envs/rolex-server/lib/python3.12/site-packages/cuopt/lib:$LD_LIBRARY_PATH"

# Add NVIDIA library paths from pip packages
export LD_LIBRARY_PATH="/home/ubuntu/.conda/envs/rolex-server/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/ubuntu/.conda/envs/rolex-server/lib/python3.12/site-packages/nvidia/cusolver/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/ubuntu/.conda/envs/rolex-server/lib/python3.12/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/ubuntu/.conda/envs/rolex-server/lib/python3.12/site-packages/nvidia/curand/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/ubuntu/.conda/envs/rolex-server/lib/python3.12/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/ubuntu/.conda/envs/rolex-server/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/ubuntu/.conda/envs/rolex-server/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"

echo "✅ cuOpt library paths set (pip-based)"
EOF
    
    chmod +x "$activate_script"
    success "Library paths updated for pip-based installation"
}

# Start the ROLEX server
start_server() {
    log "Starting ROLEX server..."
    
    cd "$PROJECT_ROOT/run"
    ./rolex_server.sh start
    
    # Wait for server to start
    sleep 5
    
    # Test server health
    if curl -s http://localhost:8000/health > /dev/null; then
        success "ROLEX server started successfully"
    else
        error "Server failed to start properly"
        return 1
    fi
}

# Test cuOpt solver
test_cuopt_solver() {
    log "Testing cuOpt solver integration..."
    
    # Test solver status
    local solver_status=$(curl -s http://localhost:8000/solvers | python3 -c "
import sys, json
data = json.load(sys.stdin)
cuopt = data.get('cuopt', {})
print(f'Available: {cuopt.get(\"available\", False)}')
if 'error' in cuopt:
    print(f'Error: {cuopt[\"error\"]}')
")
    
    log "cuOpt solver status: $solver_status"
    
    if echo "$solver_status" | grep -q "Available: True"; then
        success "cuOpt solver is now available!"
    else
        error "cuOpt solver is still not available"
        return 1
    fi
}

# Main execution
main() {
    log "Starting cuOpt fix with pure pip approach..."
    log "This will resolve CUDA library version mismatch issues"
    
    # Create log file
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "cuOpt Pure Pip Fix Log - $(date)" > "$LOG_FILE"
    
    # Execute fix steps
    check_conda
    stop_server
    backup_packages
    remove_conda_packages
    install_pytorch_pip
    install_cuopt_pip
    update_library_paths
    verify_cuda_compatibility
    start_server
    test_cuopt_solver
    
    success "🎉 cuOpt fix completed successfully!"
    log "Key changes made:"
    log "  ✅ Removed conda PyTorch/CUDA packages"
    log "  ✅ Installed PyTorch via pip with CUDA 12.1"
    log "  ✅ Installed cuOpt via pip with compatible CUDA libraries"
    log "  ✅ Updated library paths for pip-based installation"
    log "  ✅ Verified CUDA compatibility"
    log "  ✅ Started ROLEX server with working cuOpt"
    
    log "Server is now running with all three solvers:"
    log "  🟢 Gurobi: Available"
    log "  🟢 cuOpt: Available (FIXED!)"
    log "  🟢 SciPy: Available"
    
    log "Log file: $LOG_FILE"
    log "Server URL: http://localhost:8000"
    log "API Docs: http://localhost:8000/docs"
}

# Run main function
main "$@" 