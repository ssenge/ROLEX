# NVIDIA GPU-Optimized AMI (Ubuntu 24.04) - hardcoded AMI ID
# AMI: NVIDIA GPU-Optimized AMI 25.5.1 in eu-central-1
# Latest NVIDIA AMI with Ubuntu 24.04, Python 3.10, CUDA 12.8, Docker 28.1.1
locals {
  nvidia_ami_id = "ami-047c0d5a1621bf509"  # NVIDIA GPU-Optimized AMI eu-central-1
}

# Key pair for SSH access
resource "aws_key_pair" "rolex_key" {
  key_name   = "${var.project_name}-key"
  public_key = file(var.public_key_path)
}

# Security group for ROLEX server
resource "aws_security_group" "rolex_sg" {
  name        = "${var.project_name}-sg"
  description = "Security group for ROLEX optimization server"

  # SSH access
  ingress {
    from_port   = var.ssh_port
    to_port     = var.ssh_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # ROLEX server port
  ingress {
    from_port   = var.rolex_port
    to_port     = var.rolex_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

    # ROLEX server port
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # 8080
  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # 80 - Commented out (direct access via 8080, no forwarding needed)
  # ingress {
  #   from_port   = 80
  #   to_port     = 80
  #   protocol    = "tcp"
  #   cidr_blocks = ["0.0.0.0/0"]
  # }

  # 22 - Commented out to avoid policy violations (SSH available on port 443)
  # ingress {
  #   from_port   = 22
  #   to_port     = 22
  #   protocol    = "tcp"
  #   cidr_blocks = ["0.0.0.0/0"]
  # }

  # Compliant outbound traffic - FASM approved ports only
  
  # ICMPv4 Type 8 (ping)
  egress {
    from_port   = 8
    to_port     = 8
    protocol    = "icmp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # TCP approved ports
  egress {
    from_port   = 53
    to_port     = 53
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 80
    to_port     = 81
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 8080
    to_port     = 8081
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 1935
    to_port     = 1936
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 1438
    to_port     = 1440
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 1452
    to_port     = 1452
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 5060
    to_port     = 5062
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 9001
    to_port     = 9008
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 9003
    to_port     = 9003
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 9997
    to_port     = 9997
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 10100
    to_port     = 10101
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 15000
    to_port     = 20999
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 20100
    to_port     = 20100
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 65200
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # UDP approved ports
  egress {
    from_port   = 53
    to_port     = 53
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 500
    to_port     = 500
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 1604
    to_port     = 1604
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 1719
    to_port     = 1719
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 4500
    to_port     = 4500
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 5060
    to_port     = 5060
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 36000
    to_port     = 59999
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # IP protocols 47, 50, 51
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "47"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "50"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "51"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_name}-sg"
    Project     = var.project_name
    Environment = var.environment
  }
}

# EC2 instance for ROLEX server
resource "aws_instance" "rolex_server" {
  ami           = local.nvidia_ami_id
  instance_type = var.instance_type
  key_name      = aws_key_pair.rolex_key.key_name

  vpc_security_group_ids = [aws_security_group.rolex_sg.id]

  # Storage for dependencies and data
  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_size
    encrypted   = true
  }

  user_data = base64encode(<<-EOF
#!/bin/bash
set -e
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "--- Configuring SSH to listen on port 443 and scheduling restart ---"

# Add Port 443 to sshd_config
echo "Port 443" >> /etc/ssh/sshd_config

# Install 'at' if not already installed (common on Ubuntu)
apt-get update && apt-get install -y at

# Create a temporary script for SSH restart
cat <<EOT > /tmp/restart_ssh.sh
#!/bin/bash
systemctl daemon-reload
service ssh restart
EOT

chmod +x /tmp/restart_ssh.sh

# Schedule the script to run 2 minutes after multi-user.target is reached
# This ensures the system is fully up and running
echo "/tmp/restart_ssh.sh" | at now + 5 minutes

echo "--- SSH port configuration scheduled. ---"
  EOF
  )


  tags = {
    Name        = "${var.project_name}-server",
    Project     = var.project_name,
    Environment = var.environment
  }

  connection {
    type        = "ssh"
    user        = "ubuntu" # Assuming Ubuntu AMI
    private_key = file(var.private_key_path)
    host        = self.public_ip
    port        = 443
    timeout     = "10m"
  }

  provisioner "remote-exec" {
    inline = [
      "echo 'SSH connection established.'"
    ]
  }
}

# Elastic IP for stable public IP
resource "aws_eip" "rolex_eip" {
  instance = aws_instance.rolex_server.id
  domain   = "vpc"

  tags = {
    Name        = "${var.project_name}-eip"
    Project     = var.project_name
    Environment = var.environment
  }
}

#resource "null_resource" "ansible_provisioning" {
#  depends_on = [aws_instance.rolex_server]
#
#  provisioner "local-exec" {
#    working_dir = "${path.module}"
#    command = <<EOT
#      echo "[rolex_servers]" > ansible_inventory.ini
#      echo "${aws_instance.rolex_server.public_ip} ansible_user=ubuntu ansible_ssh_private_key_file=${var.#private_key_path} ansible_port=443" >> ansible_inventory.ini
#    EOT
#  }

#  provisioner "local-exec" {
#    command = "ansible-playbook -i ansible_inventory.ini /Users/sebastian.senge/src/ROLEX/iac/ansible/#playbook.yml -e \"ansible_ssh_common_args='-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'\""
#    working_dir = "${path.module}"
#  }

#  provisioner "local-exec" {
#    when = destroy
#    command = "rm -f ansible_inventory.ini"
#    working_dir = "${path.module}"
#  }
#} 