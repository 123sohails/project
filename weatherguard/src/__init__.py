"""WeatherGuard package initialization."""

from .data_engine import generate_historical_dataset, get_india_grid, get_live_inference_grid
from .model_engine import WeatherGuardEngine

__all__ = [
    "generate_historical_dataset",
    "get_india_grid",
    "get_live_inference_grid",
    "WeatherGuardEngine",
]

__version__ = "1.0.0"
