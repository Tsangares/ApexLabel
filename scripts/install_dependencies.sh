#!/bin/bash
set -e

echo "=== SAM-Bootstrap Installation ==="
echo ""

# Check Python version
python3 --version || { echo "Error: Python 3 required"; exit 1; }

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create directory structure
echo "Creating directories..."
mkdir -p data/sample_images
mkdir -p output
mkdir -p models

# Download SAM weights
SAM_CHECKPOINT="models/sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_CHECKPOINT" ]; then
    echo "Downloading SAM model weights (2.4GB)..."
    wget -q --show-progress -O "$SAM_CHECKPOINT" \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    echo "SAM model downloaded to: $SAM_CHECKPOINT"
else
    echo "SAM model already exists at: $SAM_CHECKPOINT"
fi

# Check for Ollama and pull LLaVA
if command -v ollama &> /dev/null; then
    echo "Ollama found. Pulling LLaVA model..."
    ollama pull llava:7b || echo "Warning: Failed to pull LLaVA. You can pull it later with: ollama pull llava:7b"
else
    echo ""
    echo "Note: Ollama not found."
    echo "For LLaVA bootstrap training, install Ollama from: https://ollama.ai"
    echo "Then run: ollama pull llava:7b"
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Copy and customize the configuration:"
echo "   cp config/default_config.yaml my_config.yaml"
echo "   # Edit my_config.yaml and set class_names"
echo ""
echo "3. Run the annotation tool:"
echo "   python -m sam_annotation --config my_config.yaml data/sample_images"
echo ""
