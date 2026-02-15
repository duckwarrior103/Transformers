#!/usr/bin/env bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting interactive GPU session (1x A100, 4 CPU cores, 1 day)..."

echo 'Loading modules...'
module load apps/binapps/conda/miniforge3/25.9.1
module load libs/cuda/12.4.1
module load apps/binapps/pytorch/2.6.0-312-gpu-cu124

echo 'Activating virtual environment...'
source .venv/bin/activate

echo 'Starting fresh SSH agent...'
eval "$(ssh-agent -s)"

echo 'Adding SSH key...'
ssh-add ~/.ssh/id_ed25519

echo 'Testing GitHub SSH connection...'
ssh -T git@github.com

echo 'Interactive session ready.'
exec bash
