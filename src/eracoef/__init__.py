"""era-coefs: mixed-model RAPM for box-score coefficient drift across NBA eras."""
from .design import FEATURES, DesignSpec, WindowData, build_design
from .exposure import BoxExposure
from .estimator import MixedModelRAPM, MixedModelRAPMCV

__all__ = [
    "FEATURES", "DesignSpec", "WindowData", "build_design",
    "BoxExposure", "MixedModelRAPM", "MixedModelRAPMCV",
]
