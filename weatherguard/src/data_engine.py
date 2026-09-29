"""Synthetic data generation utilities for WeatherGuard.

This module creates the India-wide forecast grid used for both historical backtesting
and live NWP-style inference. ERA5 is intentionally used only in the historical
label-generation path to create a ground-truth bust target. Live inference is
strictly forecast-only and contains no ERA5-derived values.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

LAT_MIN = 8.0
LAT_MAX = 37.0
LON_MIN = 68.0
LON_MAX = 98.0
GRID_STEP = 0.25

RegionName = Literal["North", "South", "East", "West", "Central", "ALL"]


def _region_for_point(lat: float, lon: float) -> str:
    """Assign a broad meteorological region to a grid cell."""
    if lat >= 24.0:
        return "North"
    if lat <= 18.0:
        return "South"
    if lon >= 86.0:
        return "East"
    if lon <= 75.0:
        return "West"
    return "Central"


def get_india_grid(step: float = GRID_STEP) -> pd.DataFrame:
    """Generate a 0.25° x 0.25° grid covering mainland India.

    Arguments:
        step: grid resolution in degrees.

    Returns:
        A pandas DataFrame containing the aligned grid and region metadata.
    """
    lats = np.arange(LAT_MIN, LAT_MAX + step / 2.0, step)
    lons = np.arange(LON_MIN, LON_MAX + step / 2.0, step)

    grid = pd.MultiIndex.from_product([lats, lons], names=["lat", "lon"]).to_frame(index=False)
    grid["lat"] = grid["lat"].round(2)
    grid["lon"] = grid["lon"].round(2)
    grid["region"] = grid.apply(lambda row: _region_for_point(row["lat"], row["lon"]), axis=1)
    grid["grid_id"] = np.arange(len(grid))
    return grid[["grid_id", "lat", "lon", "region"]].copy()


def generate_historical_dataset(n_samples: int = 2500) -> pd.DataFrame:
    """Create a synthetic forecast dataset with ERA5-derived bust labels.

    The ERA5 reference is used strictly for target generation. This function simulates
    historical NWP forecasts and ERA5 observations so the model can learn forecast
    reliability patterns. The target is a bust label when the absolute error exceeds
    15 mm and the forecast rainfall exceeds 10 mm.
    """
    rng = np.random.default_rng(42)
    grid = get_india_grid()

    sample_rows = []
    for _ in range(max(1, int(n_samples))):
        cell = grid.sample(1, random_state=rng.integers(0, 10_000)).iloc[0]
        lat = float(cell["lat"])
        lon = float(cell["lon"])
        region = str(cell["region"])

        lead_time = int(rng.integers(1, 11))
        region_bias = {"North": 1.4, "South": 0.9, "East": 1.3, "West": 1.1, "Central": 1.0}[region]

        forecast_rain = (
            8.0
            + 0.45 * lat
            + 0.25 * (lon - 80)
            + 1.35 * lead_time
            + rng.normal(0, 4.5)
            + region_bias * 5.5
        )
        forecast_rain = max(0.0, float(forecast_rain))

        # ERA5-grounded reference used only to generate the training target.
        era5_reference = (
            forecast_rain
            + rng.normal(2.0, 5.5)
            + 0.9 * lead_time
            + np.sin((lat + lon) / 12.0) * 6.0
            + (rng.random() < 0.22) * rng.uniform(12.0, 35.0)
        )
        era5_reference = max(0.0, float(era5_reference))

        # The verification error is the literal difference from the reference field.
        error_mm = abs(era5_reference - forecast_rain)
        is_bust = bool((error_mm > 15.0) and (forecast_rain > 10.0))

        # Forecast-only feature set for the model; no ERA5 variables are kept.
        features = {
            "grid_id": int(cell["grid_id"]),
            "lat": lat,
            "lon": lon,
            "region": region,
            "lead_time": lead_time,
            "forecast_rain_mm": round(forecast_rain, 2),
            "temperature_c": round(26 + 0.18 * lat - 0.10 * lead_time + rng.normal(0, 3.2), 2),
            "relative_humidity": round(60 + 0.7 * lat + 5 * region_bias + rng.normal(0, 11), 2),
            "wind_speed_kmh": round(10 + 0.7 * lead_time + 0.5 * abs(lon - 80) + rng.normal(0, 4.5), 2),
            "surface_pressure_hpa": round(1008 - 0.12 * lat + 0.4 * lead_time + rng.normal(0, 4.0), 2),
            "dew_point_c": round(18 + 0.15 * lat + rng.normal(0, 3.0), 2),
            "convective_index": round(25 + 0.55 * lead_time + rng.normal(0, 8.0), 2),
            "u_wind": round(rng.normal(0, 7.0), 2),
            "v_wind": round(rng.normal(0, 7.0), 2),
            "cloud_cover_pct": round(30 + 0.7 * lead_time + 0.5 * forecast_rain + rng.normal(0, 12), 2),
            "forecast_bias": round(forecast_rain - era5_reference, 2),
            "era5_reference_rain_mm": round(era5_reference, 2),
            "absolute_error_mm": round(error_mm, 2),
            "is_bust": int(is_bust),
        }
        sample_rows.append(features)

    df = pd.DataFrame(sample_rows)
    return df


def get_live_inference_grid(
    lead_time: int,
    selected_region: RegionName = "ALL",
) -> pd.DataFrame:
    """Generate forecast-only feature vectors for live model inference.

    This function creates a grid for the chosen region and lead time without any ERA5
    or reference field generation. It is intentionally limited to forecast-derived
    features that a live NWP stack could emit.
    """
    if not 1 <= int(lead_time) <= 10:
        raise ValueError("lead_time must be between 1 and 10 days.")

    grid = get_india_grid()
    if selected_region != "ALL":
        region_filter = selected_region.strip().title()
        if region_filter not in {"North", "South", "East", "West", "Central"}:
            raise ValueError("selected_region must be one of: ALL, North, South, East, West, Central")
        grid = grid[grid["region"] == region_filter].copy()

    grid = grid.reset_index(drop=True)
    rng = np.random.default_rng(7 + int(lead_time))

    features = []
    for _, row in grid.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        region = str(row["region"])

        forecast_rain = (
            6.5
            + 0.35 * lat
            + 0.25 * (lon - 80)
            + 1.2 * lead_time
            + rng.normal(0, 4.0)
            + {"North": 1.5, "South": 0.9, "East": 1.3, "West": 1.1, "Central": 1.0}[region]
            * 4.0
        )
        forecast_rain = max(0.0, float(forecast_rain))

        sample = {
            "grid_id": int(row["grid_id"]),
            "lat": round(lat, 2),
            "lon": round(lon, 2),
            "region": region,
            "lead_time": int(lead_time),
            "forecast_rain_mm": round(forecast_rain, 2),
            "temperature_c": round(26.0 + 0.12 * lat - 0.09 * lead_time + rng.normal(0, 2.7), 2),
            "relative_humidity": round(58 + 0.8 * lat + 4.5 * {"North": 1.2, "South": 1.0, "East": 1.1, "West": 1.0, "Central": 1.0}[region] + rng.normal(0, 10), 2),
            "wind_speed_kmh": round(11 + 0.7 * lead_time + 0.3 * abs(lon - 80) + rng.normal(0, 4.0), 2),
            "surface_pressure_hpa": round(1007.0 - 0.12 * lat + 0.35 * lead_time + rng.normal(0, 3.8), 2),
            "dew_point_c": round(18 + 0.18 * lat + rng.normal(0, 2.7), 2),
            "convective_index": round(20 + 0.65 * lead_time + rng.normal(0, 7.5), 2),
            "u_wind": round(rng.normal(0, 7.0), 2),
            "v_wind": round(rng.normal(0, 7.0), 2),
            "cloud_cover_pct": round(30 + 0.8 * lead_time + 0.5 * forecast_rain + rng.normal(0, 12), 2),
        }
        features.append(sample)

    return pd.DataFrame(features)
