from __future__ import annotations

from tsforecast.models.base import BaseModel
from tsforecast.models.rf import RandomForestModel
from tsforecast.models.lstm import LSTMModel

try:
    from tsforecast.models.patchtst import PatchTSTModel
except ImportError:
    PatchTSTModel = None  

__all__ = ["BaseModel", "RandomForestModel", "LSTMModel", "PatchTSTModel"]
