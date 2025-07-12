# cuOpt Fix - Pure Pip Approach

## Problem Summary

The cuOpt solver was failing to import due to CUDA library version incompatibility:
- **PyTorch** (conda): Brings `nvidia-cublas-cu12` v12.9.1.4  
- **cuOpt** (pip): Brings `nvidia-cusolver-cu12` v11.7.5.82
- **Result**: cusolver v11 trying to use cublas v12 function that doesn't exist

Error: `undefined symbol: cublasSetEnvironmentMode, version libcublas.so.12`

## Solution

**Pure Pip Approach**: Remove conda PyTorch/CUDA packages and install everything via pip to ensure compatible CUDA library versions.

## Usage

1. **Upload the fix script** to your server:
   ```bash
   # Copy fix_cuopt_pure_pip.sh to /home/ubuntu/ROLEX/run/
   ```

2. **SSH to your server** and run the fix:
   ```bash
   ssh -i your-key.pem ubuntu@52.59.152.54
   cd /home/ubuntu/ROLEX/run
   ./fix_cuopt_pure_pip.sh
   ```

3. **Wait for completion** (script will show progress with colored output)

## What the Script Does

### 1. **Backup Current State**
- Exports current conda and pip package lists
- Saves to `run/conda_packages_backup.txt` and `run/pip_packages_backup.txt`

### 2. **Remove Conda PyTorch/CUDA Packages**
- Removes: `pytorch`, `torchvision`, `torchaudio`, `pytorch-cuda`
- Removes: `nvidia-cublas-cu12`, `nvidia-cusolver-cu12`, etc.

### 3. **Install PyTorch via Pip (CUDA 12.1)**
- Installs PyTorch with CUDA 12.1 support from pip
- Uses official PyTorch CUDA 12.1 wheel index

### 4. **Install cuOpt via Pip**
- Reinstalls `cuopt-cu12==25.5.1` via pip
- Brings compatible CUDA dependencies

### 5. **Update Library Paths**
- Updates activation script with pip-based library paths
- Removes old conda-specific paths

### 6. **Verify and Test**
- Tests PyTorch CUDA functionality
- Tests cuOpt import
- Starts ROLEX server
- Verifies cuOpt solver is available

## Expected Result

After running the script, all three solvers should be available:

```json
{
  "gurobi": {"available": true},
  "cuopt": {"available": true},     // <- FIXED!
  "scipy": {"available": true}
}
```

## Rollback (If Needed)

If something goes wrong, you can restore packages:

```bash
# Restore conda packages
conda activate rolex-server
conda install --file run/conda_packages_backup.txt

# Restore pip packages  
pip install -r run/pip_packages_backup.txt
```

## Log Files

- **Fix log**: `run/cuopt_fix.log`
- **Server log**: `run/rolex_server.log`

## Architecture After Fix

```
ROLEX Server (Pure Pip CUDA)
├── PyTorch (pip) → CUDA 12.1 libraries
├── cuOpt (pip) → Compatible CUDA 12.1 libraries
├── Gurobi → Working
└── SciPy → Working
```

**Key**: All CUDA libraries now come from pip with consistent versions, eliminating the conda/pip version mismatch that was causing the `cublasSetEnvironmentMode` symbol error. 