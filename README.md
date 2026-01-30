# XHEC Eleven Data Science Challenge: Plane Taxi Time Forecasting

<!-- Build & CI Status -->
![CI](https://github.com/auggy-ntn/eleven-data-challenge/actions/workflows/ci.yaml/badge.svg?event=push)

<!-- Code Quality & Tools -->
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

<!-- Environment & Package Management -->
![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

## Table of Contents

- [XHEC Eleven Data Science Challenge: Plane Taxi Time Forecasting](#xhec-eleven-data-science-challenge-plane-taxi-time-forecasting)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Quick Start](#quick-start)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Architecture](#architecture)
    - [Data Pipeline (Bronze → Silver → Gold)](#data-pipeline-bronze--silver--gold)
    - [Model Training](#model-training)
    - [Prediction \& Explainability](#prediction--explainability)
    - [Streamlit Dashboard](#streamlit-dashboard)
  - [Commands Reference](#commands-reference)
    - [Data \& Pipeline](#data--pipeline)
    - [Experiments](#experiments)
    - [Code Quality](#code-quality)
    - [Dashboard](#dashboard)
  - [Configuration](#configuration)
    - [params.yaml](#paramsyaml)
    - [Environment Variables](#environment-variables)
  - [Additional Documentation](#additional-documentation)
  - [Authors](#authors)

---

## Introduction

This repository contains the codebase for the **XHEC-Eleven Data Science Challenge** focused on forecasting aircraft taxi-out times at airports using machine learning techniques.

**Key Features:**
- **Bronze/Silver/Gold data pipeline** architecture for robust data processing
- **Multiple ML models**: Linear Regression, XGBoost, LightGBM, CatBoost, Random Forest
- **Hyperparameter optimization** with Optuna
- **Experiment tracking** with MLflow (Databricks backend)
- **Data versioning** with DVC (OVH Object Storage backend)
- **Model explainability** with SHAP values
- **Interactive dashboard** built with Streamlit

---

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/auggy-ntn/eleven-data-challenge
cd eleven-data-challenge

# 2. Install dependencies
uv sync

# 3. Set up pre-commit hooks
uv run pre-commit install

# 4. Configure environment variables (see docs/SETUP.md for details)
cp .env.example .env
# Edit .env with your credentials

# 5. Configure DVC remote
source .env
dvc remote modify --local ovh-storage access_key_id $OVH_ACCESS_KEY_ID
dvc remote modify --local ovh-storage secret_access_key $OVH_SECRET_ACCESS_KEY

# 6. Pull data from remote storage
uv run dvc pull

# 7. Launch the dashboard
uv run streamlit run src/streamlit/streamlit_app.py
```

For detailed setup instructions, see [docs/SETUP.md](docs/SETUP.md).

---

## Project Structure

```
eleven-data-challenge/
├── data/                          # Data pipeline storage
│   ├── bronze/                    # Raw, immutable source data
│   │   ├── 0. Airport data/       # Flight records
│   │   ├── 1. AC characteristics/ # Aircraft specifications
│   │   ├── 2. Weather data/       # Historical weather
│   │   └── 3. Test set/           # Test data files
│   ├── silver/                    # Cleaned, validated data (parquet)
│   ├── gold/                      # Feature-engineered, model-ready data
│   └── predictions/               # Model predictions for dashboard
│
├── src/                           # Source code
│   ├── pipelines/
│   │   ├── data/                  # Data transformation scripts
│   │   │   ├── bronze_to_silver.py
│   │   │   └── silver_to_gold.py
│   │   ├── models/                # Model training & prediction
│   │   │   ├── train_models.py
│   │   │   ├── create_predictions.py
│   │   │   ├── xgboost_model.py
│   │   │   ├── lightgbm_model.py
│   │   │   ├── catboost_model.py
│   │   │   ├── random_forest.py
│   │   │   └── linear_regression.py
│   │   └── utils/                 # Pipeline utilities
│   │       ├── model_inputs_loading.py
│   │       └── model_prediction_making.py
│   ├── streamlit/                 # Interactive dashboard
│   │   ├── streamlit_app.py       # Main app entry point
│   │   ├── components.py          # UI components (flight cards)
│   │   ├── data_loading.py        # Data loading utilities
│   │   └── looks.py               # Styling and colors
│   └── utils/
│       └── logger.py              # Loguru-based logging
│
├── constants/                     # Global constants
│   ├── paths.py                   # Path definitions
│   ├── column_names.py            # Dataset column names
│   └── feature_categories.py      # SHAP feature groupings
│
├── config/
│   └── loguru.yaml                # Logging configuration
│
├── docs/                          # Documentation
│   ├── SETUP.md                   # Developer setup guide
│   ├── DVC_WORKFLOW.md            # Data versioning workflow
│   └── PROJECT_OWNER_CHECKLIST.md # Owner onboarding
│
├── assets/                        # Static assets
│   └── eleven.png                 # Project logo
│
├── dvc.yaml                       # DVC pipeline definition
├── params.yaml                    # Model & pipeline parameters
├── pyproject.toml                 # Python project configuration
└── .env.example                   # Environment variables template
```

---

## Architecture

### Data Pipeline (Bronze → Silver → Gold)

The project uses a medallion architecture for data processing:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Bronze (Raw)              Silver (Clean)            Gold (Features)    │
│   ┌──────────┐              ┌──────────┐              ┌──────────┐       │
│   │  CSV     │    stage 1   │ Parquet  │    stage 2   │ Parquet  │       │
│   │  XLSX    │ ──────────▶  │ Validated│ ──────────▶  │ Encoded  │       │
│   │  XLS     │              │ Typed    │              │ Featured │       │
│   └──────────┘              └──────────┘              └──────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

| Layer | Purpose | Key Transformations |
|-------|---------|---------------------|
| **Bronze** | Raw, immutable data | None (source of truth) |
| **Silver** | Cleaned, validated | Type standardization, column renaming, parquet conversion |
| **Gold** | Model-ready | Feature engineering, cyclical encoding, one-hot encoding, NaN handling |

**Run the pipeline:**
```bash
uv run dvc repro bronze_to_silver   # Stage 1
uv run dvc repro silver_to_gold     # Stage 2
```

### Model Training

The project supports multiple ML models with Optuna hyperparameter optimization:

| Model | Dataset Type | NaN Handling |
|-------|--------------|--------------|
| Linear Regression | clean | Requires NaN-free data |
| Random Forest | clean | Requires NaN-free data |
| XGBoost | default | Handles NaN natively |
| LightGBM | default | Handles NaN natively |
| CatBoost | default | Handles NaN natively |

**Run training:**
```bash
uv run dvc repro train_models
```

### Prediction & Explainability

The prediction pipeline:
1. Loads the best model from MLflow (or specified run_id)
2. Samples flights from test data
3. Makes predictions
4. Computes SHAP values for explainability
5. Aggregates SHAP into categories (Weather, Traffic, Distance, Aircraft)

```bash
uv run dvc repro create_predictions
```

### Streamlit Dashboard

Interactive dashboard displaying flight predictions with SHAP-based explanations:

- **Flight cards** with stand, runway, departure time, and predicted taxi time
- **Expandable details** showing category-level SHAP contributions
- **Color-coded drivers**: Green (reduces taxi time) / Red (increases taxi time)

**Launch:**
```bash
uv run streamlit run src/streamlit/streamlit_app.py
```

---

## Commands Reference

### Data & Pipeline

See the `params.yaml` file for configurable parameters you can adjust, and which will affect pipeline behavior.

```bash
# Pull data from remote storage
uv run dvc pull

# Run full pipeline
uv run dvc repro

# Run individual stages
uv run dvc repro bronze_to_silver
uv run dvc repro silver_to_gold
uv run dvc repro train_models
uv run dvc repro create_predictions

# View pipeline DAG
uv run dvc dag

# Check pipeline status
uv run dvc status
```

### Experiments

```bash
# Run experiment with parameter override
uv run dvc exp run --set-param train_models.model_types='["xgboost"]'

# Compare parameters between experiments
uv run dvc params diff

# View all experiments
uv run dvc exp show
```

### Code Quality

```bash
# Lint code
uv run ruff check .

# Format code
uv run ruff format .

# Run pre-commit hooks manually
uv run pre-commit run --all-files
```

### Dashboard

```bash
# Launch Streamlit dashboard
uv run streamlit run src/streamlit/streamlit_app.py
```

---

## Configuration

### params.yaml

All tunable parameters are centralized in `params.yaml`:

```yaml
train_models:
  model_types:              # Models to train
    - linear_regression
    - xgboost
    - lightgbm
    - catboost
    - random_forest

datasets:                   # Dataset type per model
  linear_regression: clean  # NaN-free dataset
  xgboost: default          # Dataset with NaN (model handles it)

create_predictions:
  n_samples: 10             # Flights to sample for dashboard
  model_type: xgboost       # Model to use for predictions
  run_id: null              # MLflow run ID (null = best model)

hyperparameters:            # Optuna search spaces (null = defaults)
  linear_regression:
    fit_intercept: [true, false]
  xgboost: null
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# DVC Remote Storage (OVH Object Storage)
OVH_ACCESS_KEY_ID=your_access_key
OVH_SECRET_ACCESS_KEY=your_secret_key

# MLflow Tracking (Databricks)
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your_token
MLFLOW_EXPERIMENT_ID=your_experiment_id
```

---

## Additional Documentation

| Document | Description |
|----------|-------------|
| [docs/SETUP.md](docs/SETUP.md) | Complete developer setup guide |
| [docs/DVC_WORKFLOW.md](docs/DVC_WORKFLOW.md) | Data versioning workflow and best practices |
| [docs/PROJECT_OWNER_CHECKLIST.md](docs/PROJECT_OWNER_CHECKLIST.md) | Setup guide for project owners |

---

## Authors

**XHEC Data Science Challenge Team**

- William BELAIDI
- Grégoire BIDAULT
- Aymeric DE LONGEVIALLE
- Paul FILISETTI
- Augustin NATON
- Louis PERETIE

---
