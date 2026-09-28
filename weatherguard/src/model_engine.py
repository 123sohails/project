"""Machine learning engine for WeatherGuard reliability classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from .data_engine import generate_historical_dataset


class WeatherGuardEngine:
    """A Random Forest baseline for forecast bust classification.

    The training target is a safety-critical label: the forecast is considered a bust
    when the forecast rainfall exceeds 10 mm and the absolute error against ERA5
    reference exceeds 15 mm. The model is threshold-calibrated to maximize sensitivity
    while staying operationally safe for high-impact alerts.
    """

    def __init__(self, n_estimators: int = 300, random_state: int = 42, recall_target: float = 0.85):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.recall_target = float(recall_target)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
        )
        self.feature_columns = [
            "lat",
            "lon",
            "lead_time",
            "forecast_rain_mm",
            "temperature_c",
            "relative_humidity",
            "wind_speed_kmh",
            "surface_pressure_hpa",
            "dew_point_c",
            "convective_index",
            "u_wind",
            "v_wind",
            "cloud_cover_pct",
        ]
        self.threshold = 0.5
        self.is_fitted = False

    def fit(self, df: pd.DataFrame | None = None) -> "WeatherGuardEngine":
        """Train the model on synthetic historical NWP + ERA5 target data."""
        if df is None:
            df = generate_historical_dataset(n_samples=2500)

        features = df[self.feature_columns].copy()
        target = df["is_bust"].astype(int)

        X_train, X_valid, y_train, y_valid = train_test_split(
            features,
            target,
            test_size=0.25,
            random_state=self.random_state,
            stratify=target,
        )

        self.model.fit(X_train, y_train)
        valid_proba = self.model.predict_proba(X_valid)[:, 1]
        self._calibrate_threshold(y_valid.to_numpy(), valid_proba)
        self.is_fitted = True
        return self

    def _calibrate_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        """Tune the decision threshold to meet a safety-critical recall target."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        if len(thresholds) == 0:
            self.threshold = 0.5
            return

        # precision_recall_curve returns precision/recall arrays with an extra leading
        # entry compared to thresholds. The matching logic is therefore aligned on the
        # threshold-bearing slice to avoid shape mismatches during calibration.
        precision_for_thresholds = precision[:-1]
        recall_for_thresholds = recall[:-1]

        # We need recall >= target while preserving the highest precision possible.
        valid_mask = recall_for_thresholds >= self.recall_target
        if not np.any(valid_mask):
            self.threshold = float(thresholds[-1])
            return

        candidate_thresholds = thresholds[valid_mask]
        candidate_precision = precision_for_thresholds[valid_mask]
        best_idx = int(np.argmax(candidate_precision))
        self.threshold = float(candidate_thresholds[best_idx])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return the probability of a bust for each grid cell."""
        if not self.is_fitted:
            self.fit()
        return self.model.predict_proba(df[self.feature_columns])[:, 1]

    def predict_live(self, live_df: pd.DataFrame) -> pd.DataFrame:
        """Run live inference and enrich each forecast cell with explainability.

        Returns a DataFrame with the following fields:
            bust_probability, bust_alert, reliability_score,
            estimated_error_mm, why_unreliable.
        """
        if live_df.empty:
            return pd.DataFrame(columns=[
                "grid_id",
                "lat",
                "lon",
                "region",
                "lead_time",
                "bust_probability",
                "bust_alert",
                "reliability_score",
                "estimated_error_mm",
                "why_unreliable",
            ])

        if not self.is_fitted:
            self.fit()

        live = live_df.copy()
        live["bust_probability"] = self.predict_proba(live)
        live["bust_alert"] = (live["bust_probability"] >= self.threshold).astype(bool)

        # Reliability is essentially inverse probability of bust, scaled to 0-100.
        live["reliability_score"] = np.clip((1.0 - live["bust_probability"]) * 100.0, 0.0, 100.0)

        # Estimate expected error magnitude as a function of confidence and lead time.
        live["estimated_error_mm"] = np.clip(
            8.0 + live["bust_probability"] * 30.0 + 0.9 * live["lead_time"],
            0.0,
            100.0,
        )

        # Explainability is based on the most relevant factors for risk.
        live["why_unreliable"] = live.apply(
            self._diagnose_cell,
            axis=1,
        )

        return live[
            [
                "grid_id",
                "lat",
                "lon",
                "region",
                "lead_time",
                "bust_probability",
                "bust_alert",
                "reliability_score",
                "estimated_error_mm",
                "why_unreliable",
            ]
        ].copy()

    def _diagnose_cell(self, row: pd.Series) -> str:
        """Generate a concise, human-readable explanation for a risky forecast."""
        reasons: list[str] = []

        if float(row["forecast_rain_mm"]) > 15.0:
            reasons.append("High rainfall signal")
        if float(row["lead_time"]) >= 5:
            reasons.append("Lead-time decay")
        if float(row["relative_humidity"]) > 78.0:
            reasons.append("Moisture-rich atmosphere")
        if float(row["wind_speed_kmh"]) > 30.0:
            reasons.append("Strong wind advection")
        if float(row["convective_index"]) > 45.0:
            reasons.append("Instability surge")
        if float(row["cloud_cover_pct"]) > 80.0:
            reasons.append("Persistent cloud deck")

        if not reasons:
            return "Moderate risk: forecast is relatively stable and within expected climatology"

        # Keep the explanation concise but actionable.
        return "; ".join(reasons[:3])


def build_default_engine() -> WeatherGuardEngine:
    """Convenience function returning a fitted baseline engine."""
    engine = WeatherGuardEngine()
    engine.fit()
    return engine
