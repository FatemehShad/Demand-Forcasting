import numpy as np
import pandas as pd
import pytest

from demand_forecasting.model import DemandForecaster, evaluate, features, validate


def panel(n=60):
    rows = []
    for i, week in enumerate(pd.date_range("2020-01-06", periods=n, freq="7D")):
        for store in [1, 2]:
            rows.append(dict(record_ID=len(rows), week=week, store_id=store, sku_id=10,
                             total_price=9., base_price=10., is_featured_sku=i % 2,
                             is_display_sku=0, units_sold=store * 10 + i % 4))
    return pd.DataFrame(rows)


def test_features_do_not_read_future_targets_and_preserve_order():
    d = panel()
    past, future = d.iloc[:-8], d.iloc[-8:].sample(frac=1, random_state=2)
    original = features(past, future)
    changed = future.assign(units_sold=999999)
    pd.testing.assert_frame_equal(original, features(past, changed))
    assert original.store_id.tolist() == future.store_id.tolist()
    assert original.pair_4_mean.tolist() == [past[past.store_id == s].tail(4).units_sold.mean()
                                             for s in future.store_id]
    with pytest.raises(ValueError, match="strictly after"):
        features(past, past.tail(2))


def test_roundtrip_unknown_pair_and_input_guards(tmp_path):
    d = panel()
    model = DemandForecaster(horizon=4, iterations=10).fit(d.iloc[:-8])
    future = d.iloc[-8:].drop(columns="units_sold").copy()
    future.loc[future.index[0], "store_id"] = 999
    future.loc[future.index[0], "total_price"] = np.nan
    before = model.predict(future)
    assert np.isfinite(before).all() and (before >= 0).all()
    path = tmp_path / "model.joblib"
    model.save(path)
    np.testing.assert_allclose(before, DemandForecaster.load(path).predict(future))
    with pytest.raises(ValueError, match="after cutoff"):
        model.predict(d.head(2))
    with pytest.raises(ValueError, match="Duplicate"):
        validate(pd.concat([d, d.head(1)]), training=True)
    with pytest.raises(ValueError, match="nonnegative"):
        validate(d.assign(units_sold=-1), training=True)
    with pytest.raises(ValueError, match="Missing columns"):
        validate(d.drop(columns="sku_id"))


def test_temporal_evaluation_has_disjoint_dates():
    report, rows = evaluate(panel(), horizon=4, iterations=10)
    assert report["train_end"] < report["validation_start"]
    assert len(rows) == 8
    assert report["model"]["rmsle"] >= 0
