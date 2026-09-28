"""Scientific integrity tests for WeatherGuard.

These tests validate the core constraints required for the project:
- no ERA5 leakage into live inference,
- strict Indian grid resolution,
- valid lead time domain,
- recall-focused calibration for safety-critical bust detection.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_engine import get_india_grid, get_live_inference_grid
from src.model_engine import WeatherGuardEngine


def test_no_era5_in_live_inference():
    """Live inference must not contain ERA5-related or target fields."""
    live = get_live_inference_grid(lead_time=3, selected_region="ALL")

    forbidden_columns = {"era5_actual_rain", "abs_error", "is_bust"}
    assert forbidden_columns.isdisjoint(live.columns), (
        "Live inference contains ERA5 or target leakage columns: "
        f"{sorted(set(live.columns) & forbidden_columns)}"
    )


def test_grid_resolution():
    """The India grid must be a strict 0.25 degree by 0.25 degree lattice."""
    grid = get_india_grid()

    lats = sorted(grid["lat"].unique())
    lons = sorted(grid["lon"].unique())

    assert np.allclose(np.diff(lats), 0.25, atol=1e-9), "Latitude spacing is not 0.25°"
    assert np.allclose(np.diff(lons), 0.25, atol=1e-9), "Longitude spacing is not 0.25°"

    expected_lat_count = int((37.0 - 8.0) / 0.25) + 1
    expected_lon_count = int((98.0 - 68.0) / 0.25) + 1
    assert len(lats) == expected_lat_count, "Latitude count is incorrect"
    assert len(lons) == expected_lon_count, "Longitude count is incorrect"


def test_lead_time_range():
    """Lead time must be valid only from day 1 to day 10."""
    for lead in [1, 3, 10]:
        df = get_live_inference_grid(lead_time=lead, selected_region="ALL")
        assert df["lead_time"].nunique() == 1
        assert set(df["lead_time"].unique()).issubset({lead})

    with pytest.raises(ValueError):
        get_live_inference_grid(lead_time=0, selected_region="ALL")

    with pytest.raises(ValueError):
        get_live_inference_grid(lead_time=11, selected_region="ALL")


def test_model_high_recall_calibration():
    """The calibrated threshold should maintain high recall for safety-critical bust alerts."""
    from src.data_engine import generate_historical_dataset

    data = generate_historical_dataset(n_samples=1200)
    engine = WeatherGuardEngine(recall_target=0.80)
    engine.fit(data)

    assert engine.threshold > 0, "Decision threshold should be positive"
    assert engine.recall_target >= 0.80, "Recall target is below the required safety threshold"

    # Use a direct class probability check to confirm calibration targets recall >= 0.80.
    y_pred_proba = engine.model.predict_proba(data[engine.feature_columns])[:, 1]
    y_true = data["is_bust"].astype(int).to_numpy()
    predicted_alerts = (y_pred_proba >= engine.threshold).astype(int)
    tp = float(np.sum((predicted_alerts == 1) & (y_true == 1)))
    actual_positive = float(np.sum(y_true == 1))
    recall = tp / actual_positive if actual_positive > 0 else 0.0

    assert recall >= 0.80, f"Model recall below required threshold: recall={recall:.3f}"
