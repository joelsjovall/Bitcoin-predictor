import numpy as np
import pandas as pd
import pytest

import model as model_module
from model import FEATURE_COLUMNS, build_features, predict_price, train_model


def make_price_frame(periods: int = 220) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=periods, freq="W")
    trend = np.linspace(20_000, 30_000, periods)
    seasonal = np.sin(np.arange(periods) / 3) * 500
    prices = trend + seasonal
    pct_change = pd.Series(prices).pct_change().fillna(0) * 100
    return pd.DataFrame({"date": dates, "price": prices, "pct_change": pct_change})


def test_build_features_creates_lags_rolling_features_and_targets():
    df = make_price_frame()

    features = build_features(df)
    valid_rows = features.dropna(subset=FEATURE_COLUMNS + ["target_return"])

    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert {"target_return", "target_price", "target_date"}.issubset(features.columns)
    assert not valid_rows.empty

    row = features.iloc[60]
    expected_return = df.iloc[61]["price"] / df.iloc[60]["price"] - 1
    assert row["target_return"] == pytest.approx(expected_return)
    assert row["target_price"] == pytest.approx(df.iloc[61]["price"])


def test_train_model_returns_metrics_and_saves_model(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "MODEL_DIR", tmp_path)
    df = make_price_frame()

    trained_model, metrics = train_model(df, val_size=0.2, test_size=0.2)

    assert hasattr(trained_model, "predict")
    assert model_module._model_path("Random Forest", 1).exists()
    assert metrics["n_train"] > 0
    assert metrics["n_val"] > 0
    assert metrics["n_test"] > 0
    assert "best_params" in metrics
    for key in ["val_rmse", "mae", "rmse", "r2", "naive_mae", "naive_rmse"]:
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_predict_price_returns_positive_float(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "MODEL_DIR", tmp_path)
    df = make_price_frame()
    trained_model, _ = train_model(df, val_size=0.2, test_size=0.2)

    prediction = predict_price(df, trained_model)

    assert isinstance(prediction, float)
    assert prediction > 0


def test_feature_columns_excludes_duplicate_return():
    assert "pct_change" not in FEATURE_COLUMNS
    assert "return_1w" in FEATURE_COLUMNS
    assert "weeks_to_halving" in FEATURE_COLUMNS


def test_weeks_to_halving_counts_down_and_resets():
    dates = pd.to_datetime(["2024-04-13", "2024-04-20", "2024-04-27"])
    features = build_features(pd.DataFrame({"date": dates, "price": [100, 101, 102]}))

    assert features["weeks_to_halving"].tolist() == pytest.approx([23 / 7, 208, 207])


def test_weeks_to_halving_handles_missing_and_overdue_halvings():
    for date in ["2012-11-21", "2030-01-01"]:
        features = build_features(pd.DataFrame({"date": [date], "price": [100]}))
        assert features.loc[0, "weeks_to_halving"] == 0


def test_weeks_to_halving_does_not_use_future_halvings(monkeypatch):
    frame = make_price_frame(periods=60)
    original = build_features(frame)
    monkeypatch.setattr(model_module, "HALVINGS", model_module.HALVINGS[:-1])
    without_future_halving = build_features(frame)

    pd.testing.assert_frame_equal(
        original[FEATURE_COLUMNS], without_future_halving[FEATURE_COLUMNS]
    )
