"""MetaFiLM: metadata-conditioned FiLM modulation for cold-start demand forecasting."""

from importlib.metadata import PackageNotFoundError, version

from .forecaster import (
    MetaFiLMConfig,
    MetaFiLMForecaster,
    MetaFiLMModel,
    ModelNotFittedError,
    StabilizedFiLMLayer,
    TimeSeriesDataset,
    ValidationError,
)

try:
    __version__ = version("metafilm")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0+local"

__all__ = [
    "MetaFiLMConfig",
    "MetaFiLMForecaster",
    "MetaFiLMModel",
    "ModelNotFittedError",
    "StabilizedFiLMLayer",
    "TimeSeriesDataset",
    "ValidationError",
    "__version__",
]
