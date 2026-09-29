# WeatherGuard

WeatherGuard is an AI-based forecast reliability layer for Numerical Weather Prediction (NWP) products. It is designed for SIH 2026 Problem Statement 26079 and focuses on identifying forecast busts before they impact decision-making in weather-sensitive operations.

## Overview

The system combines:
- a synthetic India-wide 0.25° x 0.25° forecast grid,
- NWP-like forecast features for lead times Day 1 to Day 10,
- a Random Forest classifier calibrated for high recall,
- a Streamlit dashboard for operational monitoring and reliability visualization.

The model is intentionally built around a safety-first rule:
- a forecast is considered a bust when forecast rainfall is above 10 mm and the error against a reference target exceeds 15 mm,
- the detection threshold is tuned to prioritize high recall ($\ge 0.85$), which is critical for early warning and reliability monitoring.

## Scientific Rules

This repository follows a strict operational protocol:

1. ERA5 is used only in the historical backtesting/training pipeline to generate the bust target.
2. Live inference uses only forecast-derived features such as rainfall, humidity, temperature, pressure, instability indices, and winds.
3. No ERA5 fields are generated, passed, or displayed in the live inference path.
4. Reliability is treated as the inverse of bust probability, scaled to 0-100%.
5. The model calibration favors safety-critical alerts over false negatives.

## Project Structure

```text
weatherguard/
├── app.py
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── data_engine.py
    └── model_engine.py
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows PowerShell
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the dashboard

From the project root:

```bash
streamlit run app.py
```

### 4. Run the test suite

```bash
pytest
```

## Operational Workflow

1. Training and evaluation data are generated in `src/data_engine.py` using forecast-like features and ERA5-derived labels.
2. The Random Forest baseline is trained in `src/model_engine.py` and calibrated for safety-critical recall.
3. The dashboard in `app.py` visualizes:
   - reliability across the Indian grid,
   - lead-time degradation curves,
   - forecast passport cards and high-risk cell alerts.
4. The user can inspect reliability by region (North, South, East, West, Central, or ALL) and by lead time (Day 1 to Day 10).

## Prototype Evidence (Generated Run)

The dashboard's **Model Evidence** tab runs one reproducible end-to-end example from
the generated historical dataset. In the current prototype run, an Andhra-region D5
cell produced:

| Forecast rain | Reference rain | Expected error | Bust | Model probability | Reliability | Alert |
| ---: | ---: | ---: | :---: | ---: | ---: | :---: |
| 23.72 mm | 39.89 mm | 16.17 mm | YES | 48.85% | 51.1% | TRIGGERED |

The model explanation for this cell is **High rainfall signal; Lead-time decay**.
The held-out test partition for the same run produced precision **0.430**, recall
**0.884**, F1 **0.578**, PR-AUC **0.504**, and Brier score **0.224**. It contained
146 bust samples and 229 non-bust samples, with confusion matrix:

```text
                 Predicted non-bust    Predicted bust
Actual non-bust          58                 171
Actual bust              17                 129
```

These are synthetic-prototype results, not operational NWP validation results.

## How the Model Works

The baseline model includes:
- rainfall amount,
- lead time,
- humidity,
- convective instability,
- wind speed,
- temperature, pressure, cloud cover, and other NWP descriptors.

These features are used to estimate:
- bust probability,
- binary bust alert,
- reliability score,
- expected error magnitude,
- diagnostic factors explaining why the system is unreliable.

## Files

### `src/data_engine.py`
- Generates the India 0.25° x 0.25° forecast grid.
- Simulates historical forecast and ERA5 reference values.
- Produces live forecast-only evaluation vectors for day 1-10 and region-specific analysis.

### `src/model_engine.py`
- Implements `WeatherGuardEngine` with a Random Forest baseline.
- Calibrates the decision threshold using precision-recall curves.
- Returns explainable risk diagnostics for operational usage.

### `app.py`
- Hosts the complete Streamlit dashboard.
- Includes the 3 major tabs required for the SIH product demo.

## Notes for Deployment

- This repository is intentionally synthetic and demonstrates the workflow for a reliability layer.
- In production, the system should ingest real forecast fields from NWP systems and replace the synthetic data generation with live operational feature pipelines.
- ERA5 or any reference dataset should remain restricted to offline evaluation and model development, not live prediction.
