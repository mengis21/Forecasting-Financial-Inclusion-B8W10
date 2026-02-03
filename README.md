# Forecasting Financial Inclusion in Ethiopia

This repository contains the end-to-end workflow for:

- Task 1: Data exploration and enrichment
- Task 2: EDA with visualizations and insights
- Task 3: Event impact modeling (event–indicator association matrix)
- Task 4: Forecasting (2025–2027) with scenario ranges
- Task 5: Streamlit dashboard

## Project Structure
- data/raw: source datasets
- data/processed: analysis-ready data
- notebooks: analysis notebooks
- reports/figures: generated charts

## How to Run (Interim)
1. Create and activate a Python environment.
2. Install dependencies:
   - `pip install -r requirements.txt`

### Build / regenerate processed dataset

- `python src/build_dataset.py`

### Generate final-report assets (forecasts + matrix + figures)

- `python -m src.generate_assets`

Outputs:

- `models/forecast_table.csv`
- `models/event_indicator_matrix.csv`
- `reports/figures/forecast_access.png`
- `reports/figures/forecast_usage_p2p.png`
- `reports/figures/event_indicator_matrix.png`

### Notebooks

- Task 1–2 EDA: `notebooks/task1_task2_eda.ipynb`
- Task 3: `notebooks/task3_impact_modeling.ipynb`
- Task 4: `notebooks/task4_forecasting.ipynb`

### Dashboard

- `streamlit run dashboard/app.py`

## Data Sources
- Starter dataset: ethiopia_fi_unified_data.xlsx
- Reference codes: reference_codes.xlsx


