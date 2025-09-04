# ROLEX Infrastructure Configuration
# Copy this file to terraform.tfvars and modify as needed

# AWS Configuration
aws_region = "eu-central-1"

# Project Configuration
project_name = "rolex"
environment  = "dev"

# Instance Configuration
instance_type = "g5.xlarge" #"g4dn.xlarge"  # GPU instance for cuOpt support
rolex_port    = 8080

# Storage Configuration
root_volume_size = 128  # GB - NVIDIA AMI requires minimum 128GB

# SSH Configuration
public_key_path  = "~/.ssh/id_rsa.pub"
private_key_path = "~/.ssh/id_rsa"

# ROLEX Configuration
conda_env_name = "rolex-server"
rolex_repo_url = "https://github.com/ssenge/ROLEX.git"
enable_gpu     = true

# Security Configuration
allowed_cidr_blocks = ["0.0.0.0/0"]  # Restrict this in production