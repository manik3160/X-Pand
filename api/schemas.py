"""
api/schemas.py
===============
Pydantic request / response models for the BI-101 Geospatial
Profitability Predictor API.
"""

from typing import List, Optional

from pydantic import BaseModel


# ──────────────────────────────────────────────────────────────────────
# Request models
# ──────────────────────────────────────────────────────────────────────

class LocationInput(BaseModel):
    """Single location sent by the client."""
    lat: float
    lon: float
    grid_id: Optional[str] = None


class PredictRequest(BaseModel):
    """Body for /predict and /batch endpoints."""
    locations: List[LocationInput]


class OptimizeRequest(BaseModel):
    """Body for /optimize endpoint."""
    max_hubs: int = 10
    min_separation_km: float = 2.0
    min_prob_threshold: float = 0.5


# ──────────────────────────────────────────────────────────────────────
# Response models
# ──────────────────────────────────────────────────────────────────────

class SHAPDriver(BaseModel):
    """A single SHAP feature-impact pair."""
    feature: str
    impact: float


class PredictionResult(BaseModel):
    """Prediction output for one grid cell."""
    grid_id: str
    lat: float
    lon: float
    p_profit: float
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    recommendation: str
    shap_drivers: List[SHAPDriver]
    is_cold_start: bool


class PredictResponse(BaseModel):
    """Wrapper returned by /predict and /batch."""
    predictions: List[PredictionResult]


class OptimizeResponse(BaseModel):
    """Wrapper returned by /optimize."""
    selected_hubs: List[str]
    total_score: float
    separation_constraint_met: bool
