# ROLEX - Remote Optimization Library EXecution

A FastAPI-based optimization server that accepts OMMX format problems and solves them using various solvers (Gurobi, cuOpt, SciPy).

## 🚀 Quick Start

Deploy ROLEX infrastructure and server with a single command:

```bash
cd iac && ./deploy.sh deploy
```

## 📋 Prerequisites

- **Terraform** >= 1.0
- **Ansible** >= 2.9
- **AWS CLI** (configured with credentials)
- **SSH Key Pair** at `~/.ssh/id_rsa` and `~/.ssh/id_rsa.pub`
- **jq** (for JSON processing)

### Install Prerequisites

```bash
# macOS
brew install terraform ansible awscli jq

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install terraform ansible awscli jq

# Generate SSH key if needed
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Terraform     │───▶│   AWS EC2       │───▶│   Ansible       │
│   (iac/)        │    │   Infrastructure │    │   (ansible/)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Infrastructure  │    │ Dynamic         │    │ ROLEX Server    │
│ as Code         │    │ Inventory       │    │ Configuration   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Configuration

### 1. Update Repository URL

Edit `iac/terraform.tfvars`:

```hcl
# ROLEX Configuration
rolex_repo_url = "https://github.com/ssenge/ROLEX.git"
```

### 2. Configure AWS Settings

```hcl
# AWS Configuration
aws_region = "eu-central-1"
instance_type = "g4dn.xlarge"  # For GPU support
rolex_port = 8000
```

### 3. Set Environment Variables (Optional)

```bash
export TF_VAR_rolex_repo_url="https://github.com/ssenge/ROLEX.git"
export TF_VAR_instance_type="g4dn.xlarge"
export AUTO_APPROVE=true  # Skip confirmation prompts
```

## 🎯 Deployment Commands

### Full Deployment
```bash
cd iac && ./deploy.sh deploy                    # Interactive deployment
cd iac && ./deploy.sh deploy --auto-approve     # Automated deployment
```

### Step-by-Step Deployment
```bash
cd iac && ./deploy.sh infra                     # Deploy infrastructure only
cd iac && ./deploy.sh config                    # Configure servers only
cd iac && ./deploy.sh verify                    # Verify deployment
```

### Management Commands
```bash
cd iac && ./deploy.sh status                    # Show deployment status
cd iac && ./deploy.sh cleanup                   # Destroy all resources
cd iac && ./deploy.sh help                      # Show help
```

## 🏃‍♂️ Local Development

### Start Server Locally
```bash
# Activate conda environment
conda activate rolex-env

# Start server
cd server && python server.py --port 8002 &
```

### Test Client
```bash
# Test with local server
cd client && python test.py
```

## 📁 Project Structure

```
ROLEX/
├── iac/                     # Terraform infrastructure
│   ├── main.tf              # Main infrastructure
│   ├── variables.tf         # Input variables
│   ├── outputs.tf           # Output values
│   ├── versions.tf          # Provider versions
│   ├── user-data.sh         # Bootstrap script
│   ├── terraform.tfvars     # Configuration values
│   └── deploy.sh            # Main deployment script
├── ansible/                 # Ansible configuration
│   ├── terraform.py         # Dynamic inventory
│   ├── ansible.cfg          # Ansible configuration
│   ├── playbook.yml         # Main playbook
│   └── templates/
│       └── rolex.service.j2 # Systemd service template
├── server/                  # ROLEX server code
├── client/                  # ROLEX client code
└── README.md               # This file
```

## 🔍 Monitoring & Troubleshooting

### Check Deployment Status
```bash
cd iac && ./deploy.sh status
```

### View Server Logs
```bash
# Get SSH command from deployment
ssh -i ~/.ssh/id_rsa ec2-user@<PUBLIC_IP>

# View logs
sudo journalctl -u rolex -f
```

### Test Server Health
```bash
curl http://<PUBLIC_IP>:8000/health
curl http://<PUBLIC_IP>:8000/solvers
```

### Common Issues

1. **SSH Key Issues**
   ```bash
   # Generate new SSH key
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   ```

2. **AWS Authentication**
   ```bash
   # Configure AWS CLI
   aws configure
   ```

3. **Server Not Responding**
   ```bash
   # Check server logs
   ssh -i ~/.ssh/id_rsa ec2-user@<PUBLIC_IP> "sudo journalctl -u rolex -n 50"
   ```

## 🧪 Testing

### Test Deployment Pipeline
```bash
# Test infrastructure only
cd iac && ./deploy.sh infra

# Test configuration only
cd iac && ./deploy.sh config

# Verify everything works
cd iac && ./deploy.sh verify
```

### Run Client Tests
```bash
cd client
python test.py
```

## 🔒 Security

- **Firewall**: Only necessary ports (22, 8000) are open
- **SSH**: Key-based authentication only
- **Systemd**: Service runs as non-root user
- **Updates**: Automatic security updates enabled

## 📊 Supported Solvers

- **Gurobi** (with trial license)
- **cuOpt** (GPU-accelerated, NVIDIA)
- **SciPy** (fallback solver)

## 🌐 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TF_VAR_instance_type` | EC2 instance type | `g4dn.xlarge` |
| `TF_VAR_rolex_port` | Server port | `8000` |
| `TF_VAR_rolex_repo_url` | Repository URL | (required) |
| `AUTO_APPROVE` | Skip confirmations | `false` |

## 🆘 Support

1. **Check Logs**: `cd iac && ./deploy.sh status`
2. **View Documentation**: `cd iac && ./deploy.sh help`
3. **Clean Start**: `cd iac && ./deploy.sh cleanup && ./deploy.sh deploy`

## 🎉 Next Steps

After successful deployment:

1. **Test the server**: `curl http://<PUBLIC_IP>:8000/health`
2. **Run optimization**: Use client examples in `client/examples/`
3. **Monitor performance**: Check server logs and metrics
4. **Scale**: Modify `instance_type` in `terraform.tfvars` 