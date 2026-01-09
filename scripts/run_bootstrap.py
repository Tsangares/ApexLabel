#!/usr/bin/env python3
"""
Bootstrap Training CLI

Simple CLI wrapper for running bootstrap training with LLaVA validation.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ProjectConfig


def main():
    parser = argparse.ArgumentParser(
        description="Run bootstrap training with LLaVA validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_bootstrap.py --config my_config.yaml
  python scripts/run_bootstrap.py --config my_config.yaml --check-llava
        """
    )
    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--check-llava',
        action='store_true',
        help='Only check LLaVA service status'
    )
    parser.add_argument(
        '--start-llava',
        action='store_true',
        help='Start LLaVA services if not running'
    )

    args = parser.parse_args()

    # Load and validate configuration
    try:
        config = ProjectConfig.from_yaml(args.config)
        config.validate()
        print(f"Loaded configuration from: {args.config}")
        print(f"  Class names: {config.class_names}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Handle LLaVA service management
    if args.check_llava or args.start_llava:
        from bootstrap.llava_service_manager import LLaVAServiceManager
        manager = LLaVAServiceManager()

        if args.check_llava:
            manager.print_status()
            sys.exit(0)

        if args.start_llava:
            if not manager.start_all_services():
                print("Failed to start LLaVA services")
                sys.exit(1)
            print("LLaVA services started successfully")
            sys.exit(0)

    # TODO: Add actual bootstrap training logic here
    print("\nBootstrap training not yet implemented in this generalized version.")
    print("For now, use the SAM annotation tool to create training data:")
    print(f"  python -m sam_annotation --config {args.config}")


if __name__ == "__main__":
    main()
