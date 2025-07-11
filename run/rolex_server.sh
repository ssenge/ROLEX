#!/bin/bash

# ROLEX Server Management Script with Conda Environment Support
# Usage: ./rolex_server.sh [start|stop] [port] [--log-dir=path]

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVER_DIR="$PROJECT_ROOT/server"
PID_FILE="$PROJECT_ROOT/run/rolex_server.pid"

# Default settings
DEFAULT_PORT=8000
DEFAULT_LOG_DIR="."
LOG_DIR="$DEFAULT_LOG_DIR"
CONDA_ENV_NAME="rolex-server"
CONDA_PATH="/opt/miniforge"

# Check if conda is available and environment exists
check_conda_env() {
    if ! command -v conda &> /dev/null; then
        echo "⚠️  Conda not found. Make sure conda is installed and in PATH."
        return 1
    fi
    
    if ! conda info --envs | grep -q "$CONDA_ENV_NAME"; then
        echo "⚠️  Conda environment '$CONDA_ENV_NAME' not found."
        echo "   Please create it with: conda create -n $CONDA_ENV_NAME python=3.11 -y"
        return 1
    fi
    
    return 0
}

show_help() {
    echo "ROLEX Server Management Script"
    echo ""
    echo "Usage:"
    echo "  $0 start [port] [--log-dir=path]    Start the ROLEX server (default port: $DEFAULT_PORT)"
    echo "  $0 stop                             Stop the ROLEX server"
    echo "  $0 status                           Show server status"
    echo ""
    echo "Options:"
    echo "  --log-dir=path     Set log directory (default: current directory)"
    echo ""
    echo "Examples:"
    echo "  $0 start                            # Start server on port $DEFAULT_PORT, log in current dir"
    echo "  $0 start 8000                       # Start server on port 8000"
    echo "  $0 start 8080 --log-dir=/tmp        # Start server with custom log directory"
    echo "  $0 stop                             # Stop the server"
    echo "  $0 status                           # Check if server is running"
    echo ""
    echo "Requirements:"
    echo "  - Conda environment '$CONDA_ENV_NAME' must be available"
    echo "  - Python 3.11+ with required packages installed"
}

start_server() {
    # Check conda environment first
    if ! check_conda_env; then
        return 1
    fi
    
    # Parse arguments for --log-dir
    local args=("$@")
    local port
    local i=0
    
    while [[ $i -lt ${#args[@]} ]]; do
        case "${args[$i]}" in
            --log-dir=*)
                LOG_DIR="${args[$i]#*=}"
                # Remove this argument
                unset args[$i]
                args=("${args[@]}")
                ;;
            *)
                ((i++))
                ;;
        esac
    done
    
    # Extract port from remaining args
    if [ ${#args[@]} -gt 0 ]; then
        port="${args[0]}"
    fi
    port=${port:-$DEFAULT_PORT}
    
    # Set log file path (make it absolute and clean)
    if [[ "$LOG_DIR" == /* ]]; then
        # LOG_DIR is already absolute
        LOG_FILE="$LOG_DIR/rolex_server.log"
    else
        # LOG_DIR is relative, make it absolute based on current working directory
        if [ "$LOG_DIR" = "." ]; then
            LOG_FILE="$(pwd)/rolex_server.log"
        else
            LOG_FILE="$(cd "$LOG_DIR" && pwd)/rolex_server.log"
        fi
    fi
    
    # Create log directory if it doesn't exist
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR" || {
            echo "❌ Failed to create log directory: $LOG_DIR"
            return 1
        }
    fi
    
    # Check if server is already running
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p $pid > /dev/null 2>&1; then
            echo "❌ ROLEX server is already running (PID: $pid)"
            echo "   Use '$0 stop' to stop it first"
            return 1
        else
            # Remove stale PID file
            rm -f "$PID_FILE"
        fi
    fi
    
    # Check if port is available
    if lsof -i :$port > /dev/null 2>&1; then
        echo "❌ Port $port is already in use"
        echo "   Choose a different port or stop the process using port $port"
        return 1
    fi
    
    # Ensure server directory exists
    if [ ! -d "$SERVER_DIR" ]; then
        echo "❌ Server directory not found: $SERVER_DIR"
        return 1
    fi
    
    # Start the server with conda environment
    echo "🚀 Starting ROLEX server on port $port..."
    echo "🐍 Using conda environment: $CONDA_ENV_NAME"
    cd "$SERVER_DIR"
    
    # Start server with conda environment activation
    nohup bash -c "
        # Set up paths explicitly for conda
        export PATH='${CONDA_PATH}/condabin:\$PATH'
        
        # Source conda setup explicitly
        if [ -f '${CONDA_PATH}/etc/profile.d/conda.sh' ]; then
            source '${CONDA_PATH}/etc/profile.d/conda.sh'
        fi
        
        # Activate conda environment
        conda activate $CONDA_ENV_NAME || {
            echo 'Failed to activate conda environment $CONDA_ENV_NAME' >&2
            exit 1
        }
        
        # Change to server directory and start
        cd '$SERVER_DIR' || {
            echo 'Failed to change to server directory $SERVER_DIR' >&2
            exit 1
        }
        
        # Start the server
        exec python server.py --port $port
    " > "$LOG_FILE" 2>&1 &
    
    local pid=$!
    echo $pid > "$PID_FILE"
    
    # Wait a moment and check if server started successfully
    sleep 3
    if ps -p $pid > /dev/null 2>&1; then
        echo "✅ ROLEX server started successfully (PID: $pid)"
        echo "   Server URL: http://localhost:$port"
        echo "   API Docs: http://localhost:$port/docs"
        echo "   Health: http://localhost:$port/health"
        echo "   Log file: $LOG_FILE"
        echo "   Conda environment: $CONDA_ENV_NAME"
    else
        echo "❌ Failed to start ROLEX server"
        echo "   Check log file: $LOG_FILE"
        echo "   Last 10 lines of log:"
        tail -10 "$LOG_FILE" 2>/dev/null || echo "   No log content available"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop_server() {
    if [ ! -f "$PID_FILE" ]; then
        echo "❌ ROLEX server is not running (no PID file found)"
        return 1
    fi
    
    local pid=$(cat "$PID_FILE")
    
    if ! ps -p $pid > /dev/null 2>&1; then
        echo "❌ ROLEX server is not running (process $pid not found)"
        rm -f "$PID_FILE"
        return 1
    fi
    
    echo "🛑 Stopping ROLEX server (PID: $pid)..."
    
    # Try graceful shutdown first
    kill $pid
    
    # Wait for graceful shutdown
    local count=0
    while ps -p $pid > /dev/null 2>&1 && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done
    
    # Force kill if still running
    if ps -p $pid > /dev/null 2>&1; then
        echo "   Force killing server..."
        kill -9 $pid
        sleep 1
    fi
    
    if ps -p $pid > /dev/null 2>&1; then
        echo "❌ Failed to stop ROLEX server"
        return 1
    else
        echo "✅ ROLEX server stopped successfully"
        rm -f "$PID_FILE"
    fi
}

show_status() {
    # Set log file path (make it absolute and clean)
    if [[ "$LOG_DIR" == /* ]]; then
        # LOG_DIR is already absolute
        LOG_FILE="$LOG_DIR/rolex_server.log"
    else
        # LOG_DIR is relative, make it absolute based on current working directory
        if [ "$LOG_DIR" = "." ]; then
            LOG_FILE="$(pwd)/rolex_server.log"
        else
            LOG_FILE="$(cd "$LOG_DIR" && pwd)/rolex_server.log"
        fi
    fi
    
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p $pid > /dev/null 2>&1; then
            echo "✅ ROLEX server is running (PID: $pid)"
            
            # Try to get port from process (convert service names to port numbers)
            local port_info=$(lsof -n -p $pid | grep LISTEN | head -1 | awk '{print $9}' | sed 's/.*://')
            if [[ "$port_info" =~ ^[0-9]+$ ]]; then
                local port=$port_info
            else
                # Convert service name to port number using getent
                local port=$(getent services "$port_info" 2>/dev/null | awk '{print $2}' | cut -d/ -f1)
                if [ -z "$port" ]; then
                    # Fallback for common service names if getent fails
                    case "$port_info" in
                        "irdmi") port="8000" ;;
                        "http-alt") port="8080" ;;
                        *) port="$port_info" ;;
                    esac
                fi
            fi
            if [ -n "$port" ]; then
                echo "   Server URL: http://localhost:$port"
                echo "   API Docs: http://localhost:$port/docs"
                echo "   Health: http://localhost:$port/health"
            fi
            
            echo "   Log file: $LOG_FILE"
        else
            echo "❌ ROLEX server is not running (stale PID file)"
            rm -f "$PID_FILE"
        fi
    else
        echo "❌ ROLEX server is not running"
    fi
}

# Main script logic
case "${1:-}" in
    start)
        shift
        start_server "$@"
        ;;
    stop)
        stop_server
        ;;
    status)
        shift
        # Parse --log-dir for status
        while [[ $# -gt 0 ]]; do
            case $1 in
                --log-dir=*)
                    LOG_DIR="${1#*=}"
                    shift
                    ;;
                *)
                    shift
                    ;;
            esac
        done
        show_status
        ;;
    restart)
        shift
        stop_server
        sleep 2
        start_server "$@"
        ;;
    *)
        show_help
        ;;
esac 