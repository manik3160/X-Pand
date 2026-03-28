# BI-101: Geospatial Profitability Predictor — X-Pand

**X-Pand** is an end-to-end geospatial profitability prediction system built for **Tomato**, a food delivery company. It predicts whether opening a new service hub in a target area will be profitable within 6 months by analyzing historical delivery zone data over a 500m × 500m geospatial grid. The system combines spatial feature engineering, geographically weighted regression, LightGBM classification with SHAP-based interpretability, Thompson Sampling for exploration-exploitation trade-offs, and Binary Integer Programming for optimal hub placement — all served through a FastAPI backend and interactive Streamlit dashboard with pydeck map visualizations.

---

## How to Run

### 1. Install Dependencies

```bash
cd X-Pand
pip install -r requirements.txt
```

### 2. Run Notebooks (in order)

Execute notebooks sequentially to process data, engineer features, train, and evaluate the model:

```bash
jupyter notebook notebooks/
```

| Order | Notebook | Purpose |
|-------|----------|---------|
| 01 | `01_data_preparation.ipynb` | Load raw data, build geospatial grid, merge demographics |
| 02 | `02_feature_engineering.ipynb` | Compute spatial lags, density features, GWR coefficients |
| 03 | `03_model_training.ipynb` | Train LightGBM with SMOTE, Optuna tuning, SHAP analysis |
| 04 | `04_evaluation_and_optimization.ipynb` | Evaluate model, run Thompson Sampling, solve BIP for hub placement |

### 3. Start the API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API endpoints:
- `POST /predict` — Predict profitability for a single grid cell
- `POST /predict/batch` — Score up to 10,000 grid cells in a single request
- `GET /health` — Health check

### 4. Launch the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Architecture Overview

The `src/` package contains **9 core modules** organized by pipeline stage:

| Module | Role |
|--------|------|
| `src/grid_builder.py` | Constructs the 500m × 500m geospatial H3/hexagonal grid over the target region using Shapely and GeoPandas. Assigns each cell a unique `grid_id`. |
| `src/feature_engineer.py` | Computes spatial features: population density, competitor proximity, road density, order velocity, and spatial lag variables using libpysal weights matrices. |
| `src/spatial_model.py` | Runs Geographically Weighted Regression (GWR) via the `mgwr` package to capture spatially varying relationships and extracts local coefficients as features. |
| `src/classifier.py` | Trains a LightGBM binary classifier with SMOTE oversampling (imbalanced-learn), Optuna hyperparameter tuning, and calibrated probability outputs. |
| `src/explainer.py` | Generates SHAP-based interpretability — global feature importance, local force plots, and spatial SHAP value maps for every prediction. |
| `src/thompson_sampler.py` | Implements Thompson Sampling with Beta distributions (scipy.stats) for exploration-exploitation scoring of unvisited grid cells. |
| `src/optimizer.py` | Solves the Binary Integer Programming (BIP) problem via PuLP to select the optimal set of hub locations subject to budget, distance, and coverage constraints. |
| `src/predictor.py` | Orchestrates the full inference pipeline — takes raw grid coordinates, runs feature engineering, classification, confidence intervals, and returns structured predictions. |
| `src/data_loader.py` | Handles all data I/O — loads raw CSVs, validates schemas, merges delivery zone and demographic data, and exports processed DataFrames with `grid_id` integrity checks. |

---

## Performance Targets

| Metric | Target |
|--------|--------|
| **F1-Score** | > 0.80 on held-out test set |
| **Batch Scoring** | 10,000 grid cells scored in < 5 minutes |
| **BIP Optimization** | Optimal hub selection computed in < 60 seconds |
| **Confidence Intervals** | 95% CI for every probability prediction |
| **Interpretability** | Full SHAP explanations for every prediction |

---

## Project Structure

```
X-Pand/
├── data/
│   ├── raw/
│   │   ├── delivery_zones.csv
│   │   └── demographics.csv
│   └── processed/
├── models/
├── notebooks/
├── api/
│   └── __init__.py
├── app/
├── src/
│   └── __init__.py
├── requirements.txt
└── README.md
```

---

## License

Built for **Tomato** — Hackathon Submission 2026.
