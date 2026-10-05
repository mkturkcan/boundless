"""Boundless: photorealistic synthetic data for object detection in urban streetscapes (Python API)."""

from .client import CaptureResult, Client, Pose, BoundlessError
from .labels import iter_frames, load_frame

__all__ = ["CaptureResult", "Client", "Pose", "BoundlessError", "iter_frames", "load_frame"]
__version__ = "1.0.0"
