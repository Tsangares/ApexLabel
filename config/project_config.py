#!/usr/bin/env python3
"""
Central configuration class for SAM-Bootstrap projects.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
import os


@dataclass
class ProjectConfig:
    """
    Central configuration for annotation and bootstrap training.

    Users must specify class_names - there are no defaults.
    """

    # REQUIRED: Class configuration (no defaults - user must specify)
    class_names: List[str] = field(default_factory=list)

    # LLaVA prompt configuration
    llava_prompt_file: str = "prompts/validation_template.txt"
    target_description: str = ""  # Description for LLaVA prompts
    positive_indicators: List[str] = field(default_factory=list)
    negative_indicators: List[str] = field(default_factory=list)

    # Directory paths (relative to project root by default)
    data_dir: str = "./data"
    output_dir: str = "./output"
    models_dir: str = "./models"

    # YOLO training parameters
    default_epochs: int = 50
    default_batch_size: int = 8
    default_image_size: int = 640

    # Ollama/LLaVA configuration
    ollama_ports: List[int] = field(default_factory=lambda: [11434, 11435])
    llava_model: str = "llava:7b"

    # GPU configuration
    sam_device: str = "cuda:0"
    yolo_device: str = "cuda:0"

    @classmethod
    def from_yaml(cls, path: str) -> "ProjectConfig":
        """Load configuration from a YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        return cls(**filtered_data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "class_names": self.class_names,
            "llava_prompt_file": self.llava_prompt_file,
            "target_description": self.target_description,
            "positive_indicators": self.positive_indicators,
            "negative_indicators": self.negative_indicators,
            "data_dir": self.data_dir,
            "output_dir": self.output_dir,
            "models_dir": self.models_dir,
            "default_epochs": self.default_epochs,
            "default_batch_size": self.default_batch_size,
            "default_image_size": self.default_image_size,
            "ollama_ports": self.ollama_ports,
            "llava_model": self.llava_model,
            "sam_device": self.sam_device,
            "yolo_device": self.yolo_device,
        }

        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def validate(self) -> None:
        """
        Validate that required configuration is present.
        Raises ValueError if configuration is invalid.
        """
        if not self.class_names:
            raise ValueError(
                "class_names must be specified in configuration. "
                "Example: class_names: ['vehicle', 'building']"
            )

        if not all(isinstance(name, str) and name.strip() for name in self.class_names):
            raise ValueError("All class names must be non-empty strings")

    def get_absolute_path(self, relative_path: str, base_dir: Optional[str] = None) -> Path:
        """Convert a relative path to absolute, using base_dir or cwd."""
        path = Path(relative_path)
        if path.is_absolute():
            return path

        if base_dir:
            return Path(base_dir) / path
        return Path.cwd() / path

    def ensure_directories(self, base_dir: Optional[str] = None) -> None:
        """Create output and model directories if they don't exist."""
        for dir_attr in ['data_dir', 'output_dir', 'models_dir']:
            dir_path = self.get_absolute_path(getattr(self, dir_attr), base_dir)
            dir_path.mkdir(parents=True, exist_ok=True)

    @property
    def primary_class(self) -> str:
        """Get the first (primary) class name."""
        if self.class_names:
            return self.class_names[0]
        return ""

    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return len(self.class_names)


def load_config(config_path: Optional[str] = None) -> ProjectConfig:
    """
    Load configuration from file or create default.

    Args:
        config_path: Path to YAML config file. If None, looks for
                    config.yaml in current directory.

    Returns:
        ProjectConfig instance
    """
    if config_path is None:
        # Look for config in standard locations
        search_paths = [
            "config.yaml",
            "config/config.yaml",
            "config/default_config.yaml",
        ]
        for path in search_paths:
            if Path(path).exists():
                config_path = path
                break

    if config_path and Path(config_path).exists():
        return ProjectConfig.from_yaml(config_path)

    # Return empty config - user must configure
    return ProjectConfig()
