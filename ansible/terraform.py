#!/usr/bin/env python3
"""
Dynamic Ansible inventory script that reads Terraform outputs.
This script allows Ansible to automatically discover infrastructure
provisioned by Terraform without manual IP management.
"""

import json
import subprocess
import sys
import os
import argparse
from pathlib import Path

class TerraformInventory:
    def __init__(self, terraform_dir=None):
        self.terraform_dir = terraform_dir or self._find_terraform_dir()
        
    def _find_terraform_dir(self):
        """Find the Terraform directory relative to this script."""
        script_dir = Path(__file__).parent
        terraform_dir = script_dir.parent / "iac"
        
        if not terraform_dir.exists():
            raise FileNotFoundError(f"Terraform directory not found at {terraform_dir}")
        
        return str(terraform_dir)
    
    def _run_terraform_command(self, command):
        """Run a terraform command and return the output."""
        try:
            result = subprocess.run(
                command,
                cwd=self.terraform_dir,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error running terraform command: {e}", file=sys.stderr)
            print(f"Command: {' '.join(command)}", file=sys.stderr)
            print(f"Error output: {e.stderr}", file=sys.stderr)
            sys.exit(1)
    
    def get_terraform_outputs(self):
        """Get Terraform outputs as a dictionary."""
        try:
            output = self._run_terraform_command(['terraform', 'output', '-json'])
            return json.loads(output)
        except json.JSONDecodeError as e:
            print(f"Error parsing Terraform output JSON: {e}", file=sys.stderr)
            sys.exit(1)
    
    def generate_inventory(self):
        """Generate Ansible inventory from Terraform outputs."""
        try:
            tf_outputs = self.get_terraform_outputs()
            
            # Check if ansible_inventory output exists
            if 'ansible_inventory' in tf_outputs:
                return tf_outputs['ansible_inventory']['value']
            
            # Fallback: generate inventory from individual outputs
            if 'instance_public_ip' not in tf_outputs:
                return {"_meta": {"hostvars": {}}}
            
            public_ip = tf_outputs['instance_public_ip']['value']
            private_ip = tf_outputs.get('instance_private_ip', {}).get('value', '')
            
            inventory = {
                'rolex_servers': {
                    'hosts': {
                        public_ip: {
                            'ansible_host': public_ip,
                            'ansible_user': 'ec2-user',
                            'ansible_ssh_private_key_file': '~/.ssh/id_rsa',
                            'private_ip': private_ip
                        }
                    },
                    'vars': {
                        'ansible_ssh_common_args': '-o StrictHostKeyChecking=no',
                        'ansible_python_interpreter': '/usr/bin/python3'
                    }
                },
                '_meta': {
                    'hostvars': {
                        public_ip: {
                            'ansible_host': public_ip,
                            'ansible_user': 'ec2-user',
                            'private_ip': private_ip
                        }
                    }
                }
            }
            
            return inventory
            
        except Exception as e:
            print(f"Error generating inventory: {e}", file=sys.stderr)
            return {"_meta": {"hostvars": {}}}
    
    def list_hosts(self):
        """List all hosts (for --list option)."""
        return self.generate_inventory()
    
    def get_host_vars(self, hostname):
        """Get variables for a specific host (for --host option)."""
        inventory = self.generate_inventory()
        
        # Look for the host in all groups
        for group_name, group_data in inventory.items():
            if group_name == '_meta':
                continue
            
            if 'hosts' in group_data and hostname in group_data['hosts']:
                return group_data['hosts'][hostname]
        
        # Check _meta hostvars
        if '_meta' in inventory and 'hostvars' in inventory['_meta']:
            return inventory['_meta']['hostvars'].get(hostname, {})
        
        return {}

def main():
    parser = argparse.ArgumentParser(description='Terraform dynamic inventory for Ansible')
    parser.add_argument('--list', action='store_true', help='List all hosts')
    parser.add_argument('--host', help='Get variables for a specific host')
    parser.add_argument('--terraform-dir', help='Path to Terraform directory')
    
    args = parser.parse_args()
    
    try:
        inventory = TerraformInventory(args.terraform_dir)
        
        if args.list:
            print(json.dumps(inventory.list_hosts(), indent=2))
        elif args.host:
            print(json.dumps(inventory.get_host_vars(args.host), indent=2))
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 