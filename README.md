# ROLEX - Remote Optimization Library EXecution

A FastAPI-based optimization server with Python client library and command-line interface for solving optimization problems using OMMX format.

## Features

- **FastAPI REST API** - Modern, async optimization server
- **OMMX Format Support** - Industry-standard optimization problem format
- **Multiple Solvers** - Gurobi, cuOpt (GPU-accelerated)
- **Python Client Library** - Easy-to-use Python API
- **Command-Line Interface** - CLI for file-based optimization
- **Asynchronous Processing** - Non-blocking job management

## Quick Start

### Server
```bash
cd server
pip install -r requirements.txt
python server.py
```

### Client Library
```bash
cd client
pip install -r requirements.txt
python test.py
```

### CLI
```bash
python run/rolex.py client/example.lp
```

## Installation

### Server Setup
1. **Create environment:**
   ```bash
   conda create -n rolex-env python=3.11
   conda activate rolex-env
   ```

2. **Install dependencies:**
   ```bash
   cd server
   pip install -r requirements.txt
   ```

3. **Optional: Setup Gurobi**
   - Install Gurobi Optimizer
   - Configure license (academic/commercial)
   - Test: `python -c "import gurobipy; print('Gurobi OK')"`

### Client Setup
```bash
cd client
pip install -r requirements.txt
```

## Usage

### Server API

Start the server:
```bash
cd server
python server.py
```

The server runs on `http://localhost:8000` with these endpoints:
- `GET /health` - Health check
- `GET /` - Server information
- `GET /solvers` - Available solvers
- `POST /jobs/submit` - Submit optimization job
- `GET /jobs/{job_id}` - Get job status and results

API documentation available at: `http://localhost:8000/docs`

### Client Library

```python
import client
import ommx.v1 as ommx

# Create optimization problem
x1 = ommx.DecisionVariable.continuous(id=1, name="x1", lower=0, upper=10)
x2 = ommx.DecisionVariable.continuous(id=2, name="x2", lower=0, upper=10)

objective = x1 + x2
constraint = (x1 + x2 <= 1).add_name("sum_constraint")

instance = ommx.Instance.from_components(
    decision_variables=[x1, x2],
    objective=objective,
    constraints=[constraint],
    sense=ommx.Instance.MAXIMIZE,
)

# Submit to ROLEX
rolex_client = client.Client()
problem = client.Problem.from_instance(instance)
job_id = rolex_client.submit(problem)
result = rolex_client.poll(job_id, problem)

print(f"Status: {result.status}")
print(f"Objective: {result.objective_value}")
print(f"Variables: {result.get_variables()}")
```

### CLI Interface

**Basic Usage:**
```bash
# Solve LP file
python run/rolex.py client/example.lp

# Solve with specific solver
python run/rolex.py client/example.lp --solver gurobi

# Show variable assignments
python run/rolex.py client/example.lp --show-vars
```

**Advanced Usage:**
```bash
# Python module with parameters
python run/rolex.py --python "examples/cli/advanced_problem.py:knapsack_problem(capacity=20, items=6)"

# JSON output to file
python run/rolex.py client/example.lp --format json --output results.json

# Custom server
python run/rolex.py client/example.lp --server-url http://remote-server:8000
```

## Examples

### CLI Examples
- **Simple Problems**: `examples/cli/simple_problem.py`
- **Advanced Problems**: `examples/cli/advanced_problem.py`

### Client Examples
- **Basic Usage**: `client/examples/simple_example.py`
- **File Loading**: `client/examples/lp_example.py`, `client/examples/mps_example.py`
- **Converter**: `client/examples/converter_example.py`

## Project Structure

```
ROLEX/
├── client/                    # Python client library
│   ├── requirements.txt      # Client dependencies
│   ├── client.py            # Main client library
│   ├── test.py              # Test client
│   ├── cli/                 # CLI support modules
│   ├── examples/            # Client examples
│   └── test_files/          # Test data
├── server/                   # FastAPI server
│   ├── requirements.txt     # Server dependencies
│   ├── server.py            # Main server
│   ├── models.py            # Data models
│   ├── job_manager.py       # Job management
│   └── solvers/             # Solver implementations
├── examples/
│   └── cli/                 # CLI examples
├── run/
│   └── rolex.py            # CLI entry point
└── README.md               # This file
```

## Troubleshooting

### Common Issues

**Server won't start:**
- Check port 8000 is available
- Verify all dependencies installed
- Check Python version (3.11+ required)

**Gurobi not available:**
- Check license validity: `gurobi_cl --license`
- Verify installation: `python -c "import gurobipy"`
- Check environment variables

**Client connection errors:**
- Verify server is running: `curl http://localhost:8000/health`
- Check firewall settings
- Verify server URL in client

**Import errors:**
- Check Python environment
- Verify all requirements installed
- Check PYTHONPATH if using custom locations

### Getting Help

1. Check server logs for error messages
2. Test with simple examples first
3. Verify solver availability via `/solvers` endpoint
4. Use `--verbose` flag in CLI for detailed output

## Development

### Running Tests
```bash
# Server tests
cd server
python -m pytest

# Client tests
cd client
python test.py
```

### Adding New Solvers
1. Create solver class in `server/solvers/`
2. Inherit from `BaseSolver`
3. Implement required methods
4. Register in `JobManager`

### API Documentation
Interactive API docs available at `http://localhost:8000/docs` when server is running.

## License

MIT License - See LICENSE file for details. 