"""Detector registry. Add new detectors here."""

from .base import Detector
from .tmr import TMRDetector
from .desklib import DesklibDetector
from .fast_detect import FastDetectDetector
from .binoculars import BinocularsDetector
from .editlens import EditLensDetector
from .radar import RadarDetector

DETECTORS: dict[str, type[Detector]] = {
    "tmr": TMRDetector,
    "desklib": DesklibDetector,
    "fast-detect": FastDetectDetector,
    "binoculars": BinocularsDetector,
    "editlens": EditLensDetector,
    "radar": RadarDetector,
}


def get_detector(name: str) -> Detector:
    """Instantiate a detector by name."""
    if name not in DETECTORS:
        raise ValueError(f"Unknown detector: {name}. Available: {list(DETECTORS)}")
    return DETECTORS[name]()
