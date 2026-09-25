import numpy as np
import pandas as pd
import pytest

import model as model_module
from macro import MACRO_FEATURE_COLUMNS, build_macro_features, fetch_macro_prices
from model import FEATURE_COLUMNS, build_features, predict_price, train_model


def make_frames(periods=900):
    dates = pd.date_range("2008-01-06", periods=periods, freq="W")
    bitcoin = pd.DataFrame({"date": dates, "price": np.linspace(100, 30000, periods)})
    macro = pd.DataFrame({
        "date": dates - pd.Timedelta(days=2),
        "sp500": np.linspace(1000, 4000, periods),
        "gold": np.linspace(500, 2000, periods),
        "treasury_10y": np.linspace(2, 5, periods),
    })
    return bitcoin, macro


def test_macro_units_and_bitcoin_features_unchanged():
    bitcoin, macro = make_frames()
    combined = build_features(bitcoin, macro_df=macro)
    baseline = build_features(bitcoin)
    pd.testing.assert_frame_equal(combined[FEATURE_COLUMNS], baseline[FEATURE_COLUMNS])
    for asset in ("sp500", "gold"):
        for weeks in (12, 52):
            assert combined.loc[60, f"{asset}_return_{weeks}w"] == pytest.approx(
                macro.loc[60, asset] / macro.loc[60 - weeks, asset] - 1
            )
    assert combined.loc[60, "treasury_10y_level"] == macro.loc[60, "treasury_10y"]
    assert combined.loc[60, "treasury_10y_change_12w"] == pytest.approx(
        macro.loc[60, "treasury_10y"] - macro.loc[48, "treasury_10y"]
    )


@pytest.mark.parametrize("bitcoin_unit,macro_unit", [("us", "s"), ("ns", "us"), ("s", "ns")])
def test_macro_accepts_different_datetime_resolutions(bitcoin_unit, macro_unit):
    bitcoin, macro = make_frames(100)
    expected = build_features(bitcoin, macro_df=macro)
    bitcoin["date"] = bitcoin["date"].astype(f"datetime64[{bitcoin_unit}]")
    macro["date"] = macro["date"].astype(f"datetime64[{macro_unit}]")
    actual = build_features(bitcoin, macro_df=macro)
    pd.testing.assert_frame_equal(actual[FEATURE_COLUMNS + MACRO_FEATURE_COLUMNS], expected[FEATURE_COLUMNS + MACRO_FEATURE_COLUMNS])


def test_macro_never_uses_future_or_same_day_prices():
    bitcoin, macro = make_frames(100)
    cutoff = bitcoin.loc[70, "date"]
    original = build_macro_features(bitcoin.date, macro)
    altered = macro.copy()
    altered.loc[altered.date >= cutoff, ["sp500", "gold", "treasury_10y"]] *= 10
    same_day = pd.DataFrame({"date": [cutoff], "sp500": [99999], "gold": [99999], "treasury_10y": [99]})
    changed = build_macro_features(bitcoin.date, pd.concat([altered, same_day]))
    pd.testing.assert_frame_equal(original.iloc[:71], changed.iloc[:71])


def test_missing_macro_not_filled_from_future_or_stale_values():
    bitcoin, macro = make_frames(100)
    missing = macro.drop(index=[60, 61])
    features = build_macro_features(bitcoin.date, missing)
    assert features.loc[60:61, "treasury_10y_level"].isna().all()
    assert np.isnan(features.loc[72, "gold_return_12w"])
    truncated = build_macro_features(bitcoin.date, macro.iloc[10:])
    assert truncated.loc[:9, "treasury_10y_level"].isna().all()


@pytest.mark.parametrize("horizon", model_module.FORECAST_HORIZONS.values())
@pytest.mark.parametrize("method", ["Linjär regression", "Ridge"])
def test_macro_training_prediction_and_separate_files(tmp_path, monkeypatch, horizon, method):
    monkeypatch.setattr(model_module, "MODEL_DIR", tmp_path)
    bitcoin, macro = make_frames()
    baseline_path = model_module._model_path(method, horizon)
    baseline_path.write_bytes(b"unchanged")
    trained, metrics = train_model(bitcoin, method=method, horizon_weeks=horizon, macro_df=macro)
    assert list(trained.feature_names_in_) == FEATURE_COLUMNS + MACRO_FEATURE_COLUMNS
    assert model_module._model_path(method, horizon, True).exists()
    assert baseline_path.read_bytes() == b"unchanged"
    assert np.isfinite(metrics["return_rmse_pct"])
    complete = build_features(bitcoin, horizon, macro).dropna(subset=FEATURE_COLUMNS + MACRO_FEATURE_COLUMNS + ["target_return"])
    complete = model_module._training_window(complete, metrics["history_weeks"])
    assert metrics["n_production"] == len(complete)
    assert trained.named_steps["standardscaler"].n_samples_seen_ == len(complete)
    assert metrics["production_target_end"] == bitcoin.date.max()
    assert metrics["production_train_end"] == bitcoin.date.max() - pd.Timedelta(weeks=horizon)
    assert metrics["data_start"] == bitcoin.date.min()
    assert metrics["data_end"] == bitcoin.date.max()
    assert metrics["test_target_end"] == metrics["test_end"] + pd.Timedelta(weeks=horizon)
    prediction = predict_price(bitcoin, trained, horizon_weeks=horizon, macro_df=macro)
    assert np.isfinite(prediction)
    with pytest.raises(ValueError, match="aktuella data"):
        predict_price(bitcoin, trained, horizon_weeks=horizon, macro_df=macro.iloc[:-2])


def test_empty_macro_history_has_clear_training_error():
    bitcoin, macro = make_frames()
    with pytest.raises(ValueError, match="För lite komplett historik"):
        train_model(bitcoin, macro_df=macro.iloc[:0])


def test_download_normalizes_multiindex_and_excludes_today(monkeypatch):
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    raw = pd.DataFrame(
        [4.0, 4.1, 99.0], index=pd.date_range(today - pd.Timedelta(days=2), periods=3, name="Date"),
        columns=pd.MultiIndex.from_tuples([("Close", "ticker")]),
    )
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: raw.copy())
    result = fetch_macro_prices("2020-01-01")
    assert len(result) == 2
    assert result.treasury_10y.tolist() == [4.0, 4.1]


def test_empty_download_is_reported(monkeypatch):
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: pd.DataFrame())
    with pytest.raises(RuntimeError, match="Inga makrodata"):
        fetch_macro_prices("2020-01-01")


@pytest.mark.parametrize("download_fails", [False, True])
def test_app_keeps_bitcoin_chart_and_handles_macro_result(tmp_path, monkeypatch, download_fails):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    bitcoin, macro = make_frames(220)
    bitcoin["date"] = bitcoin["date"].astype("datetime64[us]")
    macro["date"] = macro["date"].astype("datetime64[s]")
    monkeypatch.setattr(model_module, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(model_module, "METHODS", {"Random Forest": model_module.METHODS["Linjär regression"]})
    monkeypatch.setitem(model_module.PARAM_GRIDS, "Random Forest", [{}])
    monkeypatch.setattr("db.load_prices_df", lambda: bitcoin)
    # Appens automatiska uppdatering får inte skriva till den riktiga databasen.
    monkeypatch.setattr("db.ingest_live", lambda: 0)
    daily_price = pd.DataFrame({"Close": [30_000.0]})
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: daily_price.copy())

    def download(start):
        if download_fails:
            raise RuntimeError("Test: offline")
        return macro

    monkeypatch.setattr("macro.fetch_macro_prices", download)
    st.cache_data.clear()
    st.cache_resource.clear()
    try:
        app = AppTest.from_file("../app.py")
        app.run(timeout=30)
        assert not app.exception
        assert any(metric.value == "30,000" for metric in app.metric)
        assert not any("avkastning i procentenheter" in metric.label for metric in app.metric)
        status_messages = [element.value for element in [*app.success, *app.info]]
        assert not any("🏆" in message for message in status_messages)
        expected_rankings = 1 if download_fails else 2
        assert sum("bästa metoden för" in message.lower() for message in status_messages) == expected_rankings
        assert len(app.get("plotly_chart")) == (1 if download_fails else 2)
        if download_fails:
            assert any("Test: offline" in warning.value for warning in app.warning)
        else:
            assert any("Makroprognos" in metric.label for metric in app.metric)
            labels = [metric.label for metric in app.metric]
            assert labels.count("RMSE (hela sluttestet, USD)") == 2

        retrain = next(button for button in app.button if button.label == "Träna om modellen")
        retrain.click().run(timeout=30)
        assert not app.exception
        assert len(app.get("plotly_chart")) == (1 if download_fails else 2)
    finally:
        st.cache_data.clear()
        st.cache_resource.clear()
