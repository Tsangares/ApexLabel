"""
SAM Annotation Tool - Interactive image annotation with Segment Anything Model

A GUI tool for annotating images using SAM with manual bounding box fallback.
"""

__version__ = "1.0.0"
__author__ = "SAM Annotation Team"

from .sam_annotator import SAMAnnotator, main

__all__ = ["SAMAnnotator", "main"]