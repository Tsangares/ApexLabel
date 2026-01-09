# LLaVA Prompt Templates

This directory contains prompt templates for LLaVA-based validation during bootstrap training.

## Template Placeholders

The `validation_template.txt` file uses the following placeholders that should be configured in your `config.yaml`:

| Placeholder | Config Key | Description |
|------------|------------|-------------|
| `{TARGET_DESCRIPTION}` | `target_description` | Description of what to detect |
| `{POSITIVE_INDICATORS}` | `positive_indicators` | List of visual cues that suggest the object is present |
| `{NEGATIVE_INDICATORS}` | `negative_indicators` | List of visual cues that suggest the object is NOT present |

## Example Configuration

For vehicle detection:

```yaml
target_description: "a vehicle such as a car, truck, or bus"

positive_indicators:
  - "Rectangular or elongated shape with clear edges"
  - "Visible wheels or wheel wells"
  - "Located on roads, parking lots, or driveways"
  - "Typical vehicle colors (white, gray, black, silver)"
  - "Length-to-width ratio typical of vehicles"

negative_indicators:
  - "Irregular or organic shapes"
  - "Building features (rooftops, HVAC units)"
  - "Vegetation or tree canopies"
  - "Too small or too large to be a vehicle"
  - "Excessive blur or darkness"
```

For solar panel detection:

```yaml
target_description: "a solar panel installation on a rooftop"

positive_indicators:
  - "Rectangular grid pattern"
  - "Dark blue or black color"
  - "Located on building rooftops"
  - "Uniform reflective surface"
  - "Multiple panels in array formation"

negative_indicators:
  - "Windows or skylights"
  - "Dark roofing materials"
  - "Shadows from trees or structures"
  - "Water features or pools"
```

## Response Format

LLaVA should respond with JSON containing:

```json
{
  "is_target_object": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Explanation of decision",
  "object_type": "detected type or null",
  "image_quality": "clear|blurry|partial|unclear",
  "visual_cues": ["list", "of", "cues"]
}
```

## Custom Prompts

You can create custom prompt files and reference them in your config:

```yaml
llava_prompt_file: "prompts/my_custom_prompt.txt"
```

The prompt file should include the same placeholders for automatic substitution,
or you can hardcode the values directly in your custom prompt.
