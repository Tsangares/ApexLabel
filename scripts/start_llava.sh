#!/bin/bash
# Start LLaVA services for bootstrap training
# This script manages Ollama services for dual-GPU processing

set -e

cd "$(dirname "$0")/.."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama not found. Install from: https://ollama.ai"
    exit 1
fi

# Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run the LLaVA service manager
python -m bootstrap.llava_service_manager "$@"
