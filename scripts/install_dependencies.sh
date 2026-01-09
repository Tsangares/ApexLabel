#!/bin/bash
# ApexLabel Installation Script
# Re-runnable - safe to execute multiple times

set -e
cd "$(dirname "$0")/.."

echo ""
echo "    ___                   __          __         __"
echo "   /   |  ____  ___  _  _/ /   ____ _/ /_  ___  / /"
echo "  / /| | / __ \/ _ \| |/_/ /   / __ \`/ __ \/ _ \/ /"
echo " / ___ |/ /_/ /  __/>  </ /___/ /_/ / /_/ /  __/ /"
echo "/_/  |_/ .___/\___/_/|_/_____/\__,_/_.___/\___/_/"
echo "      /_/"
echo ""
echo "=== Installation ==="
echo ""

# Check Python version
python3 --version || { echo "Error: Python 3.8+ required"; exit 1; }

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv .venv
else
    echo "[1/4] Virtual environment exists"
fi

echo "[2/4] Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip and install dependencies
echo "[3/4] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Create directory structure
mkdir -p data/sample_images output models

# Download SAM weights
echo "[4/4] Checking SAM model..."
SAM_CHECKPOINT="models/sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_CHECKPOINT" ]; then
    echo "      Downloading SAM weights (2.4GB)..."
    wget -q --show-progress -O "$SAM_CHECKPOINT" \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
else
    echo "      SAM model exists"
fi

# Check for Ollama
echo ""
if command -v ollama &> /dev/null; then
    echo "[Optional] Ollama found - pulling LLaVA..."
    ollama pull llava:7b 2>/dev/null || echo "         LLaVA pull skipped (run manually: ollama pull llava:7b)"
else
    echo "[Optional] Ollama not installed (needed for bootstrap training)"
    echo "         Install from: https://ollama.ai"
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Quick Start:"
echo "------------"
echo ""
echo "  # 1. Activate environment"
echo "  source .venv/bin/activate"
echo ""
echo "  # 2. Download sample satellite images (optional)"
echo "  python scripts/download_satellite.py --lat 34.0522 --lon -118.2437 -n 25"
echo ""
echo "  # 3. Create config"
echo "  cp config/default_config.yaml config.yaml"
echo "  # Edit config.yaml and set: class_names: [\"vehicle\"]"
echo ""
echo "  # 4. Launch ApexLabel"
echo "  python -m sam_annotation --config config.yaml data/sample_images"
echo ""
echo "Sample Locations:"
echo "-----------------"
echo "  Los Angeles (ports):    --lat 33.7398 --lon -118.2605"
echo "  LA Industrial:          --lat 34.0522 --lon -118.2437"
echo "  San Francisco (docks):  --lat 37.8044 --lon -122.4200"
echo "  New York (JFK area):    --lat 40.6501 --lon -73.7845"
echo ""
