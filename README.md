# EURUSD ICT ML System

A modular framework for EURUSD trading research combining ICT features, machine learning model development, backtesting, and inference.

## Structure

- `config/` - environment, paths, and YAML configuration files
- `data/` - raw, interim, feature, label, and training data folders
- `notebooks/` - exploratory and research notebooks
- `src/` - project source code modules for ingestion, preprocessing, feature engineering, labeling, modeling, backtesting, API, and utilities
- `model_store/` - trained models and scalers
- `tests/` - unit and integration tests
- `scripts/` - lightweight orchestration scripts

## Quick Start

1. Create a Python environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Run training or API service from `scripts/`.
