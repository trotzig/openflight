"""
Streaming mode I/Q processing with 2D CFAR detection.

This module provides real-time FFT processing of continuous I/Q data
from the radar, using CFAR (Constant False Alarm Rate) detection to
adaptively threshold based on local noise.
"""

from .cfar import CFAR2DDetector, CFARConfig
from .processor import StreamingConfig, StreamingIQProcessor, StreamingSpeedDetector

__all__ = [
    "CFAR2DDetector",
    "CFARConfig",
    "StreamingConfig",
    "StreamingIQProcessor",
    "StreamingSpeedDetector",
]
