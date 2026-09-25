"""UI regression tests with isolated data sources and deterministic model results."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import model as model_module


@pytest.fixture
def app_data(monkeypatch):
    dates = pd.date_range(end=pd.Timestamp.now().normalize() - pd.Timedelta(days=3), periods=70, freq="7D")
    bitcoin = pd.DataFrame({"date": dates, "price": np.linspace(10_000, 20_000, len(dates))})
    macro = pd.DataFrame({
        "date": dates - pd.Timedelta(days=2),
        "sp500": np.linspace(3000, 4000, len(dates)),
        "gold": np.linspace(1000, 2000, len(dates)),
        "treasury_10y": np.linspace(2, 4, len(dates)),
    })
    state = SimpleNamespace(
        bitcoin=bitcoin, macro=macro, current_price=30_000.0,
        prediction=None, training_inputs=[], prediction_inputs=[],
    )

    def offline(*args, **kwargs):
        raise RuntimeError("Test: offline")

    def download(*args, **kwargs):
        if state.current_price is None:
            return pd.DataFrame()
        return pd.DataFrame({"Close": [state.current_price]})

    def train(data, method, horizon_weeks, macro_df=None):
        state.training_inputs.append(data.copy(deep=True))
        metrics = {
            "historical_margin_pct": 10.0,
            "relative_price_rmse_pct": 1.0 if method == "Ridge" else 5.0,
            "interval_coverage_pct": 75.0,
            "n_margin_calibration": 7, "n_coverage_test": 3,
            "rmse": 1.0 if method == "Random Forest" else 20.0,
            "mae": 8.0, "naive_rmse": 12.0, "naive_mae": 9.0, "r2": 0.5,
            "n_train": 20, "n_val": 10, "n_test": 10, "n_production": 40,
            "tuned": True, "val_rmse": 10.0, "best_params": {}, "history_weeks": None,
            "data_start": dates[0], "data_end": dates[-1],
            "production_train_start": dates[0], "production_train_end": dates[-5],
            "production_target_end": dates[-1],
            "test_start": dates[-14], "test_end": dates[-5],
            "test_target_start": dates[-10], "test_target_end": dates[-1],
            "test_dates": dates[-14:-4].tolist(),
        }
        return (method, macro_df is not None), metrics

    def predict(data, trained_model, horizon_weeks, macro_df=None):
        state.prediction_inputs.append(data.copy(deep=True))
        if state.prediction is not None:
            return state.prediction
        return float(data.iloc[-1]["price"]) * (1.1 if macro_df is None else 1.2)

    monkeypatch.setattr("db.load_prices_df", lambda: bitcoin.copy(deep=True))
    monkeypatch.setattr("db.ingest_csv", offline)
    monkeypatch.setattr("db.ingest_live", offline)
    monkeypatch.setattr("yfinance.download", download)
    monkeypatch.setattr("macro.fetch_macro_prices", lambda start: macro.copy(deep=True))
    monkeypatch.setattr(model_module, "train_model", train)
    monkeypatch.setattr(model_module, "predict_price", predict)
    st.cache_data.clear()
    st.cache_resource.clear()
    try:
        yield state
    finally:
        st.cache_data.clear()
        st.cache_resource.clear()


def run_app():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=15)
    assert not app.exception
    return app


@pytest.mark.parametrize("current_price", [30_000.0, None])
def test_forecast_uses_daily_price_when_available_without_changing_training(app_data, current_price):
    app_data.current_price = current_price
    original = app_data.bitcoin.copy(deep=True)
    app = run_app()
    expected_base = current_price or float(original.iloc[-1]["price"])
    expected_origin = pd.Timestamp.now().normalize() if current_price else original.iloc[-1]["date"]

    assert len(app_data.training_inputs) == 2 * len(model_module.METHODS)
    for training_data in app_data.training_inputs:
        pd.testing.assert_frame_equal(training_data, original)
    assert len(app_data.prediction_inputs) == 2
    for inference_data in app_data.prediction_inputs:
        pd.testing.assert_frame_equal(inference_data.iloc[:-1], original.iloc[:-1])
        assert inference_data.iloc[-1]["price"] == expected_base
    pd.testing.assert_frame_equal(app_data.bitcoin, original)

    charts = app.get("plotly_chart")
    assert len(charts) == 2
    for chart, multiplier in zip(charts, [1.1, 1.2]):
        forecast = json.loads(chart.proto.spec)["data"][1]
        assert forecast["y"] == pytest.approx([expected_base, expected_base * multiplier])
        assert pd.Timestamp(forecast["x"][0]) == expected_origin
        assert pd.Timestamp(forecast["x"][1]) == expected_origin + pd.Timedelta(weeks=4)

    forecasts = [metric for metric in app.metric if "prognos om" in metric.label.lower()]
    assert [metric.delta for metric in forecasts] == ["+10.0%", "+20.0%"]
    assert sum(metric.label == "Prisintervall utifrån historiska fel" for metric in app.metric) == 2
    if current_price is None:
        assert any("Kunde inte hämta dagens pris" in caption.value for caption in app.caption)


def test_method_ranking_uses_relative_error_not_usd_error(app_data):
    app = run_app()
    rankings = [message.value for message in app.info if "bästa metoden" in message.value.lower()]
    assert len(rankings) == 2
    assert all("**Ridge**" in message and "1.0 % mot 5.0 %" in message for message in rankings)
    comparison = next(table.value for table in app.dataframe if "Relativ RMSE (%)" in table.value.columns)
    assert comparison.index[0] == "Ridge"
    assert comparison["RMSE (USD)"].idxmin() == "Random Forest"


@pytest.mark.parametrize("prediction", [0.0, float("nan")])
def test_unusable_forecast_has_no_percentage_interval(app_data, prediction):
    app_data.prediction = prediction
    app = run_app()
    assert not any(metric.label == "Prisintervall utifrån historiska fel" for metric in app.metric)
    assert sum("Historisk felmarginal kan inte appliceras" in message.value for message in app.info) == 2
    charts = app.get("plotly_chart")
    assert len(charts) == 2
    expected_traces = 2 if prediction == 0 else 1
    assert all(len(json.loads(chart.proto.spec)["data"]) == expected_traces for chart in charts)
    forecasts = [metric for metric in app.metric if "prognos om" in metric.label.lower()]
    assert [metric.value for metric in forecasts] == (["0", "0 USD"] if prediction == 0 else ["Ej giltig"] * 2)
