# ApexLabel

```
    ___                   __          __         __
   /   |  ____  ___  _  _/ /   ____ _/ /_  ___  / /
  / /| | / __ \/ _ \| |/_/ /   / __ `/ __ \/ _ \/ /
 / ___ |/ /_/ /  __/>  </ /___/ /_/ / /_/ /  __/ /
/_/  |_/ .___/\___/_/|_/_____/\__,_/_.___/\___/_/
      /_/
```

Object detection annotation tool with SAM click-to-segment and integrated YOLO training.

Early-2000s inspired UI. Built for speed.

---

## What It Does

Click on objects. SAM segments them. Train YOLO. Repeat.

- **SAM Integration** — Click anywhere on an object, get a precise segmentation mask
- **YOLO Training** — Export annotations and train models without leaving the app
- **Prediction Assist** — Use your trained model to pre-label new images
- **Manual Fallback** — Draw bounding boxes when SAM isn't cooperating

---

## Getting Started

```bash
./scripts/install_dependencies.sh
source .venv/bin/activate

# Set your class names
cp config/default_config.yaml config.yaml
# Edit config.yaml → class_names: ["your_object"]

# Launch
python -m sam_annotation --config config.yaml /path/to/images
```

---

## Satellite Imagery

Need sample images? Download tiles from ArcGIS World Imagery:

```bash
# 100 tiles around Los Angeles
python scripts/download_satellite.py --lat 34.0522 --lon -118.2437 --count 100

# Higher resolution (zoom 19 = ~0.3m/pixel)
python scripts/download_satellite.py --lat 40.7128 --lon -74.0060 -n 50 -z 19

# Custom output
python scripts/download_satellite.py --lat 37.7749 --lon -122.4194 -o data/sf_tiles
```

Free for non-commercial use. Tiles download in a spiral pattern from center.

---

## Controls

| Input | Action |
|:------|:-------|
| `Click` | Segment object at cursor |
| `Shift+Click` | Delete annotation |
| `Scroll` | Adjust SAM threshold |
| `Ctrl+Scroll` | Zoom |
| `Space+Drag` | Pan |
| `←` `→` | Previous / Next image |
| `Ctrl+Z` | Undo |
| `Ctrl+S` | Save |

---

## Training Loop

1. Annotate images with SAM
2. Export → YOLO format
3. Train model (built-in)
4. Enable prediction assist
5. Annotate faster with pre-labels
6. Repeat

Best model saves to `models/current_best_yolo.pt`

---

## Configuration

```yaml
# config.yaml
class_names: ["car", "truck", "bus"]  # Required

# Optional
default_epochs: 50
default_batch_size: 8
default_image_size: 640
```

Full options in `config/default_config.yaml`

---

## Bootstrap Training (Optional)

For automated validation using LLaVA vision model:

```bash
# Install Ollama → https://ollama.ai
ollama pull llava:7b

# Configure prompts in config.yaml, then:
python scripts/run_bootstrap.py --config config.yaml --start-llava
```

---

## Requirements

- Python 3.8+
- CUDA GPU with 8GB+ VRAM
- Ollama (optional, for LLaVA bootstrap)

---

## Project Structure

```
ApexLabel/
├── sam_annotation/    # Main application
├── bootstrap/         # LLaVA validation pipeline
├── config/            # Configuration system
├── prompts/           # LLaVA prompt templates
├── scripts/           # Setup & utilities
├── data/              # Your images (gitignored)
├── models/            # Trained weights (gitignored)
└── output/            # Exports (gitignored)
```

---

## License

GPL-2.0 — See [LICENSE](LICENSE) for details.
