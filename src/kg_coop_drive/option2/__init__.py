"""Option 2 learned baseline module (isolated from existing production pipeline)."""

from .dataset import ObjectMotionDatasetBuilder, ObjectMotionTrainingExample
from .models import build_regression_backend

__all__ = [
    "ObjectMotionDatasetBuilder",
    "ObjectMotionTrainingExample",
    "build_regression_backend",
]
