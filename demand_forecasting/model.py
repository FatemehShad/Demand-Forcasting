"""Direct multiweek forecasts: no future sales or recursive predicted lags."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

KEYS = ["store_id", "sku_id"]
COLUMNS = ["record_ID", "week", *KEYS, "total_price", "base_price",
           "is_featured_sku", "is_display_sku"]


def read_data(path, *, training=False):
    """Read the supplied day/month/two-digit-year CSV schema and validate it."""
    return validate(pd.read_csv(path), training=training)


def validate(frame, *, training=False):
    d = frame.copy()
    required = COLUMNS + (["units_sold"] if training else [])
    missing = set(required) - set(d.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if d.empty:
        raise ValueError("Data is empty")
    if not pd.api.types.is_datetime64_any_dtype(d.week):
        d["week"] = pd.to_datetime(d.week, format="%d/%m/%y", errors="raise")
    if d.week.isna().any():
        raise ValueError("Missing dates")
    for c in required:
        if c != "week":
            d[c] = pd.to_numeric(d[c], errors="raise")
            if np.isinf(d[c]).any():
                raise ValueError(f"Infinite values in {c}")
            if c != "total_price" and d[c].isna().any():
                raise ValueError(f"Missing values in {c}")
    if d.record_ID.duplicated().any() or d.duplicated(["week", *KEYS]).any():
        raise ValueError("Duplicate record_ID or store/SKU/week")
    for c in ["record_ID", *KEYS]:
        if (d[c] % 1 != 0).any():
            raise ValueError(f"{c} must contain integers")
    for c in ["is_featured_sku", "is_display_sku"]:
        if not d[c].isin([0, 1]).all():
            raise ValueError(f"{c} must be binary")
    if (d.base_price <= 0).any() or (d.total_price.dropna() < 0).any():
        raise ValueError("Prices must be nonnegative and base_price positive")
    if training and (d.units_sold < 0).any():
        raise ValueError("units_sold must be nonnegative")
    return d


def features(history, future):
    """Build a single fixed-origin block using observed history only."""
    cutoff = history.week.max()
    if (future.week <= cutoff).any():
        raise ValueError("Forecast dates must be strictly after the history cutoff")
    x = future[KEYS + ["total_price", "base_price", "is_featured_sku", "is_display_sku"]].copy()
    x["total_price"] = x.total_price.fillna(x.base_price)
    x["discount"] = 1 - x.total_price / x.base_price
    x["horizon"] = (future.week - cutoff).dt.days.to_numpy() / 7
    x["time"] = (future.week - pd.Timestamp("2010-01-01")).dt.days.to_numpy() / 7
    week = future.week.dt.isocalendar().week.to_numpy(dtype=float)
    x["week_sin"] = np.sin(2 * np.pi * week / 52.18)
    x["week_cos"] = np.cos(2 * np.pi * week / 52.18)
    x["month"] = future.week.dt.month.to_numpy()
    # Merge in row order; never depend on the caller's DataFrame index.
    x = x.reset_index(drop=True)
    for keys, label in [(KEYS, "pair"), (["store_id"], "store"), (["sku_id"], "sku")]:
        for window in [4, 12, 52]:
            recent = history[history.week > cutoff - pd.Timedelta(weeks=window)]
            stats = recent.groupby(keys, observed=True).units_sold.agg(["mean", "std"])
            stats.columns = [f"{label}_{window}_{c}" for c in stats.columns]
            x = x.merge(stats, on=keys, how="left", sort=False, validate="many_to_one")
    latest = history.sort_values("week").groupby(KEYS).tail(1)
    latest = latest[KEYS + ["units_sold"]].rename(columns={"units_sold": "last_sales"})
    x = x.merge(latest, on=KEYS, how="left", sort=False, validate="many_to_one")
    # Model accepts missing group statistics for unseen pairs/stores/SKUs.
    return x.astype(float)


class DemandForecaster:
    """Train on historical horizon blocks; forecast at most `horizon` weeks ahead."""
    def __init__(self, horizon=12, iterations=180, seed=42):
        if horizon < 1 or iterations < 1:
            raise ValueError("horizon and iterations must be positive")
        self.horizon = horizon
        self.iterations = iterations
        self.seed = seed

    def fit(self, history):
        d = validate(history, training=True)
        weeks = sorted(d.week.unique())
        blocks = []
        labels = []
        # Up to six origins, each with at least 26 observed weeks.
        for end in range(len(weeks), 26 + self.horizon - 1, -self.horizon):
            start = end - self.horizon
            past = d[d.week < weeks[start]]
            target = d[d.week.isin(weeks[start:end])]
            blocks.append(features(past, target))
            labels.append(np.log1p(target.units_sold.to_numpy()))
            if len(blocks) == 6:
                break
        if not blocks:
            raise ValueError(f"Need at least {26 + self.horizon} distinct training weeks")
        self.model_ = HistGradientBoostingRegressor(
            max_iter=self.iterations, learning_rate=0.07, max_leaf_nodes=31,
            l2_regularization=2, early_stopping=False, random_state=self.seed,
        )
        self.model_.fit(pd.concat(blocks, ignore_index=True), np.concatenate(labels))
        self.history_ = d
        self.feature_names_ = list(blocks[0].columns)
        self.cutoff_ = d.week.max()
        return self

    def predict(self, future):
        if not hasattr(self, "model_"):
            raise ValueError("Call fit before predict")
        d = validate(future)
        delta = (d.week - self.cutoff_).dt.days
        if (delta <= 0).any() or (delta > 7 * self.horizon).any():
            raise ValueError(f"Forecast dates must be within {self.horizon} weeks after cutoff")
        if (delta % 7 != 0).any():
            raise ValueError("Forecast dates must follow the training weekly cadence")
        x = features(self.history_, d)[self.feature_names_]
        return np.maximum(0, np.expm1(self.model_.predict(x)))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        """Load only trusted artifacts: joblib/pickle can execute arbitrary code."""
        return joblib.load(path)


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    err = actual - predicted
    return {"mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt(np.square(err).mean())),
            "rmsle": float(np.sqrt(np.square(np.log1p(actual) - np.log1p(predicted)).mean())),
            "wape": float(np.abs(err).sum() / actual.sum()) if actual.sum() else None}


def evaluate(history, *, horizon=12, iterations=180):
    """Hold out the last entire forecast horizon, including all store/SKU pairs."""
    d = validate(history, training=True)
    weeks = sorted(d.week.unique())
    if len(weeks) < 26 + 2 * horizon:
        raise ValueError("Not enough weeks for training and temporal validation")
    cutoff = weeks[-horizon]
    train, holdout = d[d.week < cutoff], d[d.week >= cutoff]
    model = DemandForecaster(horizon=horizon, iterations=iterations).fit(train)
    predictions = model.predict(holdout)
    mean = train[train.week > train.week.max() - pd.Timedelta(weeks=12)].groupby(KEYS).units_sold.mean()
    baseline = pd.MultiIndex.from_frame(holdout[KEYS]).map(mean).to_numpy(dtype=float)
    baseline = np.where(np.isnan(baseline), train.units_sold.mean(), baseline)
    report = {"train_end": str(train.week.max().date()),
              "validation_start": str(holdout.week.min().date()),
              "validation_end": str(holdout.week.max().date()),
              "validation_rows": len(holdout), "horizon_weeks": horizon,
              "model": metrics(holdout.units_sold, predictions),
              "recent_mean_baseline": metrics(holdout.units_sold, baseline)}
    result = holdout[["record_ID", "week", *KEYS, "units_sold"]].copy()
    result["prediction"] = predictions
    result["baseline"] = baseline
    return report, result
