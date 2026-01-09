# Sample Images

Place your sample images in this directory for testing.

These files are gitignored and will not be committed to version control.

## Recommended Setup

1. Add 5-10 representative images from your dataset
2. Include both positive examples (containing target objects) and negative examples
3. Supported formats: PNG, JPG, JPEG, BMP, TIFF

## Image Requirements

- Resolution: 640x640 to 1920x1920 works best
- Clear, high-quality images produce better annotations
- For satellite imagery, use consistent zoom levels

## Getting Started

```bash
# Copy some images to this directory
cp /path/to/your/images/*.png ./

# Or download sample satellite images
# (Add your own download script or source)

# Then run the annotation tool
python -m sam_annotation --config config/default_config.yaml data/sample_images
```
