# Retail demand forecasting

A notebook and reusable Python package for the supplied weekly store/SKU sales data. Predicts `units_sold` for 12 future weeks with known prices and promotion flags. The notebook calls the package, so training and feature logic stay consistent in both workflows.

## Install and supply data

Requires Python 3.10+; validated with Python 3.12. Use the existing checkout.

```bash
cd /workspace/AI-Agent
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
```

Place `train_0irEZ2H.csv` and `test_nfaJ3J5.csv` in `data/`. The uploaded datasets are available locally but excluded from Git, along with model artifacts and prediction outputs. No API keys, external services, or GPU are required. Notebook data location can be overridden with `DEMAND_DATA_DIR`.

## Command line workflow

From the repository root with the environment activated:

```bash
# Last 12 weeks held out; writes metrics and validation predictions.
demand-forecast evaluate --train data/train_0irEZ2H.csv --output-dir artifacts
# Retrain on all observed sales and persist a model plus its history.
demand-forecast train --train data/train_0irEZ2H.csv --model artifacts/demand_model.joblib
# Predict in input order, with exactly record_ID and units_sold columns.
demand-forecast predict --model artifacts/demand_model.joblib --test data/test_nfaJ3J5.csv --output artifacts/submission.csv
pytest -q
```

`--horizon` and `--iterations` are available on train/evaluate. Defaults are 12 weeks and 180 boosting iterations. Inference rejects dates outside the trained horizon or not on the weekly cadence. Retrain with refreshed historical sales for a new forecast origin. Errors produce nonzero command exit status.

## Notebook and Python API

Open `notebooks/demand_forecasting.ipynb` in Jupyter and run all cells. It includes data checks, demand exploration, temporal evaluation, plots, SKU error analysis, model persistence, and final forecasts. An executed copy with results is in local `artifacts/`.

```python
from demand_forecasting import DemandForecaster, read_data
history = read_data("data/train_0irEZ2H.csv", training=True)
future = read_data("data/test_nfaJ3J5.csv")
model = DemandForecaster(horizon=12).fit(history)
units = model.predict(future)
model.save("artifacts/demand_model.joblib")
```

## Data contract

Both CSVs need `record_ID`, `week`, `store_id`, `sku_id`, `total_price`, `base_price`, `is_featured_sku`, `is_display_sku`; training also needs `units_sold`. Dates use `%d/%m/%y`, e.g. `16/07/13`. IDs must be numeric integers, flags binary, base prices positive, sales nonnegative. Duplicate IDs or store/SKU/week records are rejected. Missing `total_price` is allowed and filled from the row's `base_price`; other required values cannot be missing. Input need not be sorted, and prediction preserves row order. Target columns in future rows are ignored.

## Model and validation

Histogram gradient boosting predicts log sales from prices, discounts, promotion flags, calendar seasonality, horizon, store/SKU IDs, last observed sales, and 4/12/52-week historical statistics for pair, store, and SKU. Up to six historical forecast blocks create training examples. Each block freezes history at its origin; validation predicts all 12 held-out weeks without using their sales. Random train/test splitting is deliberately avoided. Early stopping is disabled to avoid a random internal validation split.

A recent 12-week pair mean is the baseline. Metrics include RMSLE (log-scale error), MAE, RMSE, and WAPE (fraction). On the supplied data, holding out April 23–July 9, 2013 (13,860 rows), model RMSLE was **0.3923** versus **0.6244** for the baseline; MAE was **13.83** versus **22.62**, RMSE **28.98** versus **45.01**, and WAPE **26.55%** versus **43.42%**. Inspect `artifacts/metrics.json` and the notebook for measured results. This is a single temporal holdout; broader rolling validation is appropriate before business deployment. Log-space fitting emphasizes relative error and can underpredict aggregate totals. No rounding, intervals, or inventory optimization is performed.

## Deployment considerations

Artifacts use joblib/pickle: load only trusted files and use compatible dependency versions. `requirements-tested.txt` records versions exercised during development. History is retained in the artifact, so apply the same access controls as to sales data. Batch inference needs no live services. Use capped thread counts for predictable resource usage.

Prices and promotions must be available at forecast time. Unknown pairs use missing statistics supported by the estimator; cold-start accuracy is unverified. Sales are a proxy for demand; stockouts and availability are not modeled. Monitor aggregate bias, SKU errors, drift, and changed data definitions, and retrain with newly observed sales. Numeric identifier splits are a pragmatic baseline rather than a learned categorical embedding.
