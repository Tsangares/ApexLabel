# SAM-Bootstrap

Interactive annotation tool with SAM (Segment Anything Model) and YOLO bootstrap training.

## Quick Start

```bash
# Install dependencies
./scripts/install_dependencies.sh

# Activate virtual environment
source .venv/bin/activate

# Configure your project
cp config/default_config.yaml my_config.yaml
# Edit my_config.yaml and set class_names: ["your_object"]

# Run the annotation tool
python -m sam_annotation --config my_config.yaml data/sample_images
```

## Configuration

Create a YAML config file with at minimum your class names:

```yaml
class_names: ["vehicle"]  # or ["car", "truck", "bus"] for multi-class
```

See `config/default_config.yaml` for all available options including:
- LLaVA prompt configuration for bootstrap validation
- YOLO training parameters (epochs, batch size, image size)
- GPU device settings
- Directory paths

## Features

### SAM Annotation Tool

- **Click-to-segment**: Click on objects and SAM automatically segments them
- **Manual mode**: Fallback to drawing bounding boxes manually
- **Threshold adjustment**: Scroll wheel to adjust segmentation sensitivity
- **YOLO export**: Export annotations to YOLO format for training
- **Built-in YOLO training**: Train YOLO models directly from the tool
- **Prediction assist**: Toggle YOLO predictions to speed up annotation

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Left Click | Segment object (SAM mode) / Start bbox (Manual mode) |
| Shift+Click | Remove annotation |
| Scroll | Adjust threshold |
| Ctrl+Scroll | Zoom in/out |
| Space+Drag | Pan image |
| Arrow Keys | Navigate images |
| Ctrl+Z | Undo last segmentation |
| Ctrl+S | Save annotations |

### LLaVA Bootstrap (Optional)

For automated bootstrap training with LLaVA validation:

1. Install Ollama: https://ollama.ai
2. Pull LLaVA model: `ollama pull llava:7b`
3. Configure your prompts in `config/default_config.yaml`
4. Run: `python scripts/run_bootstrap.py --config my_config.yaml --start-llava`

## Directory Structure

```
sam-bootstrap/
├── config/                 # Configuration files
│   ├── project_config.py   # Config class
│   └── default_config.yaml # Template config
├── sam_annotation/         # Annotation tool
├── bootstrap/              # Bootstrap training
├── prompts/                # LLaVA prompt templates
├── scripts/                # Helper scripts
├── data/                   # Your data (gitignored)
│   └── sample_images/      # Sample images
├── output/                 # Training outputs (gitignored)
└── models/                 # Model weights (gitignored)
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended for SAM)
- Ollama (optional, for LLaVA bootstrap)

## Example Workflows

### Basic Annotation

```bash
# 1. Configure
echo 'class_names: ["car"]' > my_config.yaml

# 2. Add images
cp /path/to/images/*.png data/sample_images/

# 3. Annotate
python -m sam_annotation --config my_config.yaml data/sample_images
```

### Train YOLO Model

After annotating images:
1. In the annotation tool, go to the "Training" tab
2. Click "Train YOLO Model"
3. Wait for training to complete
4. The best model is saved to `models/current_best_yolo.pt`

### Multi-class Detection

```yaml
# my_config.yaml
class_names: ["car", "truck", "bus", "motorcycle"]
```

The annotation tool will use the first class as default, but you can change the label in the UI.

## License

MIT License - see LICENSE file for details.
