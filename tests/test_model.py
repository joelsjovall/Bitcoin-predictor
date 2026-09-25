import numpy as np
import pandas as pd
import pytest

import model as model_module
from model import FEATURE_COLUMNS, build_features, predict_price, replace_latest_price_for_inference, train_model


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
    for key in ["val_rmse", "mae", "rmse", "r2", "naive_mae", "naive_rmse",
                "historical_margin_pct", "relative_price_rmse_pct", "interval_coverage_pct"]:
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_historical_margin_covers_at_least_eighty_percent_relative_to_prediction():
    predictions = np.full(10, 100.0)
    actual = np.array([100, 101, 98, 103, 96, 105, 94, 107, 70, 200])
    margin = model_module.historical_price_margin(actual, predictions)
    assert margin == pytest.approx(30)
    assert np.mean(np.abs(actual - predictions) <= predictions * margin / 100) >= 0.8


def test_interval_diagnostics_calibrates_before_checking_later_coverage():
    predictions = np.full(10, 100.0)
    actual = np.array([100, 101, 98, 103, 96, 105, 94, 120, 80, 130])
    stats = model_module.price_interval_diagnostics(actual, predictions)

    assert stats["n_margin_calibration"] == 7
    assert stats["n_coverage_test"] == 3
    assert stats["historical_margin_pct"] == pytest.approx(5)
    assert stats["relative_price_rmse_pct"] == pytest.approx(
        np.sqrt(np.mean(np.array([0, 1, -2, 3, -4, 5, -6], dtype=float) ** 2))
    )
    assert stats["interval_coverage_pct"] == 0


@pytest.mark.parametrize("predictions", [[0, 100], [-10, 100], [np.nan, 100], [np.inf, 100]])
def test_historical_margin_does_not_silently_exclude_invalid_predictions(predictions):
    assert model_module.historical_price_margin([100, 100], predictions) is None


@pytest.mark.parametrize("method", model_module.METHODS)
def test_predict_price_returns_positive_float(tmp_path, monkeypatch, method):
    monkeypatch.setattr(model_module, "MODEL_DIR", tmp_path)
    df = make_price_frame()
    trained_model, _ = train_model(df, method=method, val_size=0.2, test_size=0.2)

    prediction = predict_price(df, trained_model)

    assert isinstance(prediction, float)
    assert prediction > 0
    loaded = model_module.load_model(method)
    assert predict_price(df, loaded) == pytest.approx(prediction)


def test_latest_live_price_is_used_only_in_inference_copy():
    original = make_price_frame(60)
    original_latest_price = original.iloc[-1]["price"]
    live_price = original_latest_price * 1.1

    inference = replace_latest_price_for_inference(original, live_price)

    assert original.iloc[-1]["price"] == original_latest_price
    assert inference.iloc[-1]["price"] == pytest.approx(live_price)
    assert inference.iloc[-1]["pct_change"] == pytest.approx(
        (live_price / inference.iloc[-2]["price"] - 1) * 100
    )
    assert inference.iloc[-1]["date"] == original.iloc[-1]["date"]


def test_predict_price_uses_zero_as_floor_for_negative_raw_price():
    class BelowMinusOneReturnModel:
        def predict(self, rows):
            return np.full(len(rows), -1.5)

    prediction = predict_price(make_price_frame(60), BelowMinusOneReturnModel())

    assert prediction == 0


def test_feature_columns_excludes_duplicate_return():
    assert "pct_change" not in FEATURE_COLUMNS
    assert "return_1w" in FEATURE_COLUMNS
    assert "weeks_to_halving" in FEATURE_COLUMNS


def test_ridge_selects_alpha_on_validation_and_reloads(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "MODEL_DIR", tmp_path)
    frame = make_price_frame(900)
    features = build_features(frame, 4).dropna(subset=FEATURE_COLUMNS + ["target_return"])
    train, validation, _ = model_module._three_way_split(features, 0.2, 0.2, 4)
    scores = []
    configurations = []
    for weeks in (None, 208, 416):
        selected = train if weeks is None else train.loc[train.date > train.date.max() - pd.Timedelta(weeks=weeks)]
        for params in model_module.PARAM_GRIDS["Ridge"]:
            candidate = model_module.METHODS["Ridge"]().set_params(**params)
            candidate.fit(selected[FEATURE_COLUMNS], selected.target_return)
            prediction = validation.price.to_numpy() * (1 + candidate.predict(validation[FEATURE_COLUMNS]))
            scores.append(np.sqrt(np.mean((validation.target_price.to_numpy() - prediction) ** 2)))
            configurations.append((params, weeks))
    trained, metrics = train_model(frame, method="Ridge", horizon_weeks=4)
    expected, history = configurations[int(np.argmin(scores))]
    assert metrics["history_weeks"] == history
    assert len(metrics["validation_scores"]) == 21
    assert metrics["best_params"] == expected
    assert trained.named_steps["ridge"].alpha == expected["ridge__alpha"]
    assert metrics["val_rmse"] == pytest.approx(min(scores))
    loaded = model_module.load_model("Ridge", 4)
    assert predict_price(frame, loaded, horizon_weeks=4) == pytest.approx(
        predict_price(frame, trained, horizon_weeks=4)
    )
    expected_rows = features if history is None else features.loc[features.date > features.date.max() - pd.Timedelta(weeks=history)]
    assert metrics["n_production"] == len(expected_rows)
    assert trained.named_steps["standardscaler"].n_samples_seen_ == len(expected_rows)
    changed = frame.copy()
    changed.loc[changed.date >= metrics["test_start"], "price"] *= 3
    _, rerun = train_model(changed, method="Ridge", horizon_weeks=4)
    assert rerun["best_params"] == metrics["best_params"]
    assert rerun["history_weeks"] == history
    assert rerun["val_rmse"] == pytest.approx(metrics["val_rmse"])


def test_training_window_uses_calendar_dates_with_missing_rows():
    frame = make_price_frame(500).drop(index=range(350, 380))
    selected = model_module._training_window(frame, 208)
    assert selected.date.min() > frame.date.max() - pd.Timedelta(weeks=208)
    assert len(selected) == 178


def test_ridge_fallback_uses_all_history_and_default_alpha(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "MODEL_DIR", tmp_path)
    trained, metrics = train_model(make_price_frame(220), method="Ridge", horizon_weeks=52)
    assert not metrics["tuned"]
    assert metrics["history_weeks"] is None
    assert metrics["validation_scores"] == []
    assert trained.named_steps["ridge"].alpha == 1


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
