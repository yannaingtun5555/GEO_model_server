"""
ML Pipeline Package for Web Backend Integration (40 Targets)
"""

from .inference import ModelPipeline
from .config import TARGET_DEFINITIONS, CROPS

__all__ = ["ModelPipeline", "TARGET_DEFINITIONS", "CROPS"]
