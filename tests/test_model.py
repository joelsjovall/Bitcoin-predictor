import numpy as np
import pandas as pd
import pytest

import model as model_module
from model import FEATURE_COLUMNS, build_features, predict_next_price, train_model


def make_price_frame(periods: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=periods, freq="W")
    trend = np.linspace(20_000, 30_000, periods)
    seasonal = np.sin(np.arange(periods) / 3) * 500
    prices = trend + seasonal
    pct_change = pd.Series(prices).pct_change().fillna(0) * 100
    return pd.DataFrame({"date": dates, "price": prices, "pct_change": pct_change})


def test_build_features_creates_lags_rolling_features_and_targets():
    df = make_price_frame(12)

    features = build_features(df)
    valid_rows = features.dropna(subset=FEATURE_COLUMNS + ["target_return"])

    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert {"target_return", "target_next_price"}.issubset(features.columns)
    assert not valid_rows.empty

    row = features.iloc[5]
    expected_return = df.iloc[6]["price"] / df.iloc[5]["price"] - 1
    assert row["target_return"] == pytest.approx(expected_return)
    assert row["target_next_price"] == pytest.approx(df.iloc[6]["price"])


def test_train_model_returns_metrics_and_saves_model(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "MODEL_PATH", tmp_path / "model.pkl")
    df = make_price_frame(80)

    trained_model, metrics = train_model(df, test_size=0.25)

    assert hasattr(trained_model, "predict")
    assert model_module.MODEL_PATH.exists()
    assert metrics["n_train"] > 0
    assert metrics["n_test"] > 0
    for key in ["mae", "rmse", "r2", "naive_mae", "naive_rmse"]:
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_predict_next_price_returns_positive_float(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "MODEL_PATH", tmp_path / "model.pkl")
    df = make_price_frame(80)
    trained_model, _ = train_model(df, test_size=0.25)

    prediction = predict_next_price(df, trained_model)

    assert isinstance(prediction, float)
    assert prediction > 0
