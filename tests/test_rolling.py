import numpy as np
import pandas as pd

import model as module


def prices():
    dates = pd.date_range("2011-01-02", periods=780, freq="W")
    return pd.DataFrame({"date": dates, "price": 100 + np.arange(780) + 10 * np.sin(np.arange(780) / 9)})


def test_ridge_rolling_backtest_tunes_without_saving_models(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MODEL_DIR", tmp_path)
    frame = prices()
    cutoff = frame.date.iloc[400]
    result = module.rolling_backtest(frame, "Ridge", 4, cutoff, step_weeks=52)
    rows = result["predictions"]
    assert result["n_test"] > 1
    assert rows.tuned.any()
    assert np.isfinite(rows.predicted_price).all()
    assert (rows.train_target_end < rows.date).all()
    assert (rows.target_date < cutoff).all()
    assert not list(tmp_path.glob("*.pkl"))


def test_rolling_all_fits_precede_prediction_including_inner_validation(monkeypatch):
    df = prices()
    horizon = 52
    features = module.build_features(df, horizon)
    checks = []

    class AuditEstimator:
        def set_params(self, **params):
            return self

        def fit(self, x, y):
            self.last_target = features.loc[x.index, "target_date"].max()
            return self

        def predict(self, x):
            assert self.last_target < features.loc[x.index, "date"].min()
            checks.append(len(x))
            return np.zeros(len(x))

    monkeypatch.setitem(module.METHODS, "audit", AuditEstimator)
    monkeypatch.setitem(module.PARAM_GRIDS, "audit", [{}])
    cutoff = df.date.iloc[650]
    result = module.rolling_backtest(df, "audit", horizon, cutoff)
    rows = result["predictions"]
    assert len(checks) > len(rows) > 10  # exercises inner tuning as well as outer fits
    assert rows.target_date.max() < cutoff
    assert (rows.train_target_end < rows.date).all()
    assert (rows.n_train >= 104).all()
    assert rows.date.diff().dropna().eq(pd.Timedelta(weeks=13)).all()
    assert result["rmse"] == result["naive_rmse"]
    assert result["historical_margin_pct"] == module.historical_price_margin(rows.actual_price, rows.price)


def test_future_prices_cannot_change_earlier_rolling_predictions():
    df = prices()
    cutoff = df.date.iloc[450]
    original = module.rolling_backtest(df, "Linjär regression", 26, cutoff)
    changed = df.copy()
    changed.loc[changed.date >= cutoff, "price"] *= 100
    rerun = module.rolling_backtest(changed, "Linjär regression", 26, cutoff)
    pd.testing.assert_frame_equal(original["predictions"], rerun["predictions"])


def test_long_horizon_with_insufficient_history_returns_empty_result():
    df = prices().iloc[:400]
    result = module.rolling_backtest(df, "Linjär regression", 260, df.date.iloc[320])
    assert result["n_test"] == 0
    assert result["predictions"].empty
    assert "historical_margin_pct" not in result
