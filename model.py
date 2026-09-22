"""AI-modellering: förutspår priset ett antal veckor framåt (regression)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from db import load_prices_df
from macro import MACRO_FEATURE_COLUMNS, build_macro_features

MODEL_DIR = Path(__file__).parent
HALVINGS = pd.to_datetime(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"])
ESTIMATED_HALVING_INTERVAL_WEEKS = 208

FEATURE_COLUMNS = [
    "return_1w",
    "return_4w",
    "return_12w",
    "return_52w",
    "volatility_12w",
    "volatility_52w",
    "distance_from_ath",
    "weeks_since_halving",
    "weeks_to_halving",
    "has_halving",
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "rolling_mean_return_4",
    "rolling_std_return_4",
]

# Samma faktorer och skalning används för alla metoder.
METHODS = {
    "Linjär regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": lambda: make_pipeline(StandardScaler(), Ridge()),
    "SVR": lambda: make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.01)),
    "Random Forest": lambda: make_pipeline(
        StandardScaler(), RandomForestRegressor(n_estimators=300, random_state=42)
    ),
    "XGBoost": lambda: make_pipeline(
        StandardScaler(), XGBRegressor(n_estimators=300, random_state=42)
    ),
}

# Parametrar väljs på valideringsdata, aldrig på sluttestet.
PARAM_GRIDS = {
    "Linjär regression": [{}],
    "Ridge": [
        {"ridge__alpha": 0.1},
        {"ridge__alpha": 1.0},
        {"ridge__alpha": 10.0},
        {"ridge__alpha": 100.0},
        {"ridge__alpha": 300.0},
        {"ridge__alpha": 1000.0},
        {"ridge__alpha": 3000.0},
    ],
    "SVR": [
        {"svr__C": 1, "svr__epsilon": 0.01},
        {"svr__C": 10, "svr__epsilon": 0.01},
    ],
    "Random Forest": [
        {"randomforestregressor__n_estimators": 150, "randomforestregressor__max_depth": None},
        {"randomforestregressor__n_estimators": 300, "randomforestregressor__max_depth": 10},
    ],
    "XGBoost": [
        {"xgbregressor__n_estimators": 150, "xgbregressor__max_depth": 3},
        {"xgbregressor__n_estimators": 300, "xgbregressor__max_depth": 6},
    ],
}

FORECAST_HORIZONS = {
    "1 månad": 4,
    "3 månader": 13,
    "6 månader": 26,
    "1 år": 52,
    "2 år": 104,
    "3 år": 156,
    "5 år": 260,
}


def _model_path(method: str, horizon_weeks: int, include_macro: bool = False) -> Path:
    slug = method.lower().replace(" ", "_").replace("ä", "a").replace("ö", "o")
    prefix = "model_v4_macro_v1" if include_macro else "model_v4"
    if method == "Ridge":
        prefix += "_history_v2"
    return MODEL_DIR / f"{prefix}_{slug}_{horizon_weeks}w.pkl"


def _feature_columns(macro_df: pd.DataFrame | None) -> list[str]:
    return FEATURE_COLUMNS + (MACRO_FEATURE_COLUMNS if macro_df is not None else [])


def _fit_model(method, rows, columns, params=None):
    model = METHODS[method]()
    if params:
        model.set_params(**params)
    model.fit(rows[columns], rows["target_return"])
    return model


def build_features(
    df: pd.DataFrame,
    horizon_weeks: int = 1,
    macro_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Bygg faktorer och framtida avkastning från date och price.

    Sorterar veckodata och behåller ofullständiga rader för senaste prognosen.
    Framtida utfall används endast som mål, aldrig som faktorer.
    """
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])
    if horizon_weeks < 1 or not isinstance(horizon_weeks, int):
        raise ValueError("horizon_weeks måste vara ett positivt heltal.")
    if not out["date"].diff().dropna().eq(pd.Timedelta(weeks=1)).all():
        raise ValueError("Datan måste ha exakt en observation per vecka utan luckor.")
    if out.empty or out["price"].isna().any() or not np.isfinite(out["price"]).all() or (out["price"] <= 0).any():
        raise ValueError("Datan måste innehålla positiva, ändliga priser.")
    ret = out["price"].pct_change(fill_method=None)
    out["pct_change"] = ret * 100
    for weeks in (1, 4, 12, 52):
        out[f"return_{weeks}w"] = out["price"].pct_change(weeks, fill_method=None)
    for weeks in (12, 52):
        out[f"volatility_{weeks}w"] = ret.rolling(weeks).std()
    out["distance_from_ath"] = out["price"] / out["price"].cummax() - 1
    last_halving = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    for halving in HALVINGS:
        last_halving.loc[out["date"] >= halving] = halving
    out["has_halving"] = last_halving.notna().astype(int)
    out["weeks_since_halving"] = ((out["date"] - last_halving).dt.days / 7).fillna(0)
    out["weeks_to_halving"] = (
        ESTIMATED_HALVING_INTERVAL_WEEKS - out["weeks_since_halving"]
    ).clip(lower=0).where(out["has_halving"].eq(1), 0)
    out["ret_lag_1"] = ret.shift(1)
    out["ret_lag_2"] = ret.shift(2)
    out["ret_lag_3"] = ret.shift(3)
    out["rolling_mean_return_4"] = ret.shift(1).rolling(4).mean()
    out["rolling_std_return_4"] = ret.shift(1).rolling(4).std()
    if "volume" in out and out["volume"].notna().any():
        volume = pd.to_numeric(out["volume"], errors="coerce")
        out["volume_change_4w"] = volume.pct_change(4, fill_method=None)
        out["relative_volume_12w"] = volume / volume.rolling(12).mean().shift(1) - 1
    out["target_price"] = out["price"].shift(-horizon_weeks)
    out["target_date"] = out["date"].shift(-horizon_weeks)
    out["target_return"] = out["target_price"] / out["price"] - 1
    if macro_df is not None:
        out = out.merge(build_macro_features(out["date"], macro_df), on="date", how="left", validate="one_to_one")
    return out


MIN_EVAL_ROWS = 8  # minsta antal rader vi accepterar i en tränings-/valideringsdel


def _training_window(rows, weeks):
    """Calendar window ending at the latest eligible training example's origin.

    Features are built before filtering. A year here is 52 weeks; the horizon
    separates the latest eligible example from the current forecast date.
    """
    if weeks is None or rows.empty:
        return rows
    return rows.loc[rows["date"] > rows["date"].max() - pd.Timedelta(weeks=weeks)]


def _select_configuration(method, train, validation, columns, min_rows=MIN_EVAL_ROWS):
    # Alla metoder får samma historikalternativ så jämförelsen blir rättvis.
    windows = (None, 208, 416)
    best_params, best_window, best_score = None, None, np.inf
    scores = []
    for weeks in windows:
        selected = _training_window(train, weeks)
        if len(selected) < min_rows:
            continue
        for params in PARAM_GRIDS[method]:
            candidate = _fit_model(method, selected, columns, params)
            score = _price_rmse(validation, candidate.predict(validation[columns]))
            scores.append({"params": params.copy(), "history_weeks": weeks,
                           "n_train": len(selected), "val_rmse": score})
            if score < best_score:
                best_params, best_window, best_score = params.copy(), weeks, score
    return best_params, best_window, best_score, scores


def historical_price_margin(actual_price, predicted_price) -> float | None:
    """80:e percentilen av absoluta testfel relativt prognospriset, i procent.

    Avrunda uppåt till ett observerat fel så att minst 80 % täcks i testet.
    Icke-positiva prognoser saknar en meningsfull relativ prisfelmarginal.
    """
    actual = np.asarray(actual_price, dtype=float)
    predicted = np.asarray(predicted_price, dtype=float)
    if (actual.size == 0 or actual.shape != predicted.shape
            or not np.isfinite(actual).all() or not np.isfinite(predicted).all()
            or (predicted <= 0).any()):
        return None
    errors = np.abs(actual - predicted) / predicted * 100
    if not np.isfinite(errors).all():
        return None
    return float(np.quantile(errors, 0.8, method="higher"))


def price_interval_diagnostics(actual_price, predicted_price) -> dict:
    """Kalibrera ett 80 %-intervall och kontrollera det på senare prognoser.

    Både marginalen och procent-RMSE använder felet
    ``(utfall - prognos) / prognos``. Vid minst tio observationer används de
    äldsta 70 procenten för kalibrering och resten som en kronologiskt senare,
    fristående kontroll av intervallets täckning.
    """
    actual = np.asarray(actual_price, dtype=float)
    predicted = np.asarray(predicted_price, dtype=float)
    valid = (actual.size > 0 and actual.shape == predicted.shape
             and np.isfinite(actual).all() and np.isfinite(predicted).all()
             and (predicted > 0).all())
    if not valid:
        return {
            "historical_margin_pct": None,
            "relative_price_rmse_pct": None,
            "interval_coverage_pct": None,
            "n_margin_calibration": 0,
            "n_coverage_test": 0,
        }

    if actual.size >= 10:
        calibration_count = min(int(np.ceil(actual.size * 0.7)), actual.size - 3)
    else:
        calibration_count = actual.size
    calibration_actual = actual[:calibration_count]
    calibration_predicted = predicted[:calibration_count]
    calibration_errors = (calibration_actual - calibration_predicted) / calibration_predicted * 100
    margin = historical_price_margin(calibration_actual, calibration_predicted)

    coverage = None
    coverage_count = actual.size - calibration_count
    if coverage_count and margin is not None:
        later_errors = np.abs((actual[calibration_count:] - predicted[calibration_count:])
                              / predicted[calibration_count:] * 100)
        coverage = float(np.mean(later_errors <= margin) * 100)

    return {
        "historical_margin_pct": margin,
        "relative_price_rmse_pct": float(np.sqrt(np.mean(calibration_errors ** 2))),
        "interval_coverage_pct": coverage,
        "n_margin_calibration": calibration_count,
        "n_coverage_test": coverage_count,
    }


def _prices_from_returns(start_price, predicted_return):
    """Konvertera avkastningar till priser med 0 USD som ekonomiskt golv."""
    return np.maximum(np.asarray(start_price) * (1 + np.asarray(predicted_return)), 0.0)


def _price_rmse(rows: pd.DataFrame, pred_return: np.ndarray) -> float:
    pred_price = _prices_from_returns(rows["price"].values, pred_return)
    return float(np.sqrt(mean_squared_error(rows["target_price"].values, pred_price)))


def _three_way_split(feat: pd.DataFrame, val_size: float, test_size: float, horizon_weeks: int):
    """Delar `feat` kronologiskt i train/val/test.

    Varje fönster behöver minst `horizon_weeks` rader marginal innan nästa fönster,
    annars sträcker sig ALLA rader mål (target_date) förbi fönstrets egen gräns och
    läckage-trimningen (se nedan) tömmer det helt. Returnerar None om datan är för
    kort för det vid den här horisonten – anroparen får då falla tillbaka på en
    enklare train/test-delning.
    """
    n = len(feat)
    val_rows = max(int(n * val_size), horizon_weeks + MIN_EVAL_ROWS)
    test_rows = max(int(n * test_size), MIN_EVAL_ROWS)
    test_start = n - test_rows
    val_start = test_start - val_rows
    if val_start < horizon_weeks + MIN_EVAL_ROWS:
        return None

    train, val, test = feat.iloc[:val_start], feat.iloc[val_start:test_start], feat.iloc[test_start:]

    # Undvik läckage: ta bort träningsrader vars mål (target_date) sträcker sig in i
    # valideringsperioden, och valideringsrader vars mål sträcker sig in i testperioden.
    train = train.loc[train["target_date"] < val["date"].iloc[0]]
    val = val.loc[val["target_date"] < test["date"].iloc[0]]
    if len(train) < MIN_EVAL_ROWS or len(val) < MIN_EVAL_ROWS or len(test) < 2:
        return None
    return train, val, test


def train_model(
    df: pd.DataFrame | None = None,
    method: str = "Random Forest",
    horizon_weeks: int = 1,
    val_size: float = 0.2,
    test_size: float = 0.2,
    macro_df: pd.DataFrame | None = None,
):
    """Välj konfiguration på valideringsdata och utvärdera på ett separat sluttest.

    Slutmodellen tränas sedan på kompletta exempel inom vald historiklängd
    och sparas på disk. Returnerar modellen och utvärderingsmåtten.
    Om historiken inte räcker för validering används standardparametrar.
    """
    if method not in METHODS:
        raise ValueError(f"Okänd metod: {method}. Välj bland {list(METHODS)}")
    if df is None:
        df = load_prices_df()
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size och test_size måste vara mellan 0 och 1 och summera till mindre än 1.")

    feature_columns = _feature_columns(macro_df)
    feat = build_features(df, horizon_weeks, macro_df).dropna(subset=feature_columns + ["target_return"])
    if len(feat) < 3:
        raise ValueError("För lite komplett historik för vald modell och horisont.")

    split = _three_way_split(feat, val_size, test_size, horizon_weeks)
    history_weeks, validation_scores = None, []
    if split is not None:
        train, val, test = split
        best_params, history_weeks, best_val_rmse, validation_scores = _select_configuration(
            method, train, val, feature_columns
        )
        train_for_eval = pd.concat([train, val])
        n_train, n_val, tuned = len(_training_window(train, history_weeks)), len(val), True
    else:
        n = len(feat)
        test_rows = max(int(n * test_size), 2)
        test_start = n - test_rows
        train_for_eval, test = feat.iloc[:test_start], feat.iloc[test_start:]
        train_for_eval = train_for_eval.loc[train_for_eval["target_date"] < test["date"].iloc[0]]
        if train_for_eval.empty or len(test) < 2:
            raise ValueError("För lite historik för vald horisont.")
        best_params, best_val_rmse, n_train, n_val, tuned = None, None, len(train_for_eval), 0, False

    # Sluttest: den vinnande (eller, om tuning inte var möjlig, standard-) konfigurationen
    # tränas om på train_for_eval och utvärderas en enda gång på den helt osedda testdelen.
    train_for_eval = _training_window(train_for_eval, history_weeks)
    production_rows = _training_window(feat, history_weeks)
    eval_model = _fit_model(method, train_for_eval, feature_columns, best_params)

    pred_return = eval_model.predict(test[feature_columns])
    pred_price = _prices_from_returns(test["price"].values, pred_return)
    actual_price = test["target_price"].values
    naive_price = test["price"].values  # baseline: priset om N veckor = samma som nu

    interval_metrics = price_interval_diagnostics(actual_price, pred_price)
    metrics = {
        "method": method,
        **interval_metrics,
        "horizon_weeks": horizon_weeks,
        "tuned": tuned,
        "best_params": best_params,
        "history_weeks": history_weeks,
        "validation_scores": validation_scores,
        "n_eval_train": len(train_for_eval),
        "val_rmse": best_val_rmse,
        "mae": mean_absolute_error(actual_price, pred_price),
        "rmse": np.sqrt(mean_squared_error(actual_price, pred_price)),
        "r2": r2_score(actual_price, pred_price),
        "naive_mae": mean_absolute_error(actual_price, naive_price),
        "naive_rmse": np.sqrt(mean_squared_error(actual_price, naive_price)),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": len(test),
        "test_start": test["date"].iloc[0],
        "test_end": test["date"].iloc[-1],
        "test_dates": test["date"].tolist(),
        "test_target_start": test["target_date"].min(),
        "test_target_end": test["target_date"].max(),
        "data_start": pd.to_datetime(df["date"]).min(),
        "data_end": pd.to_datetime(df["date"]).max(),
        "production_train_start": production_rows["date"].min(),
        "production_train_end": production_rows["date"].max(),
        "production_target_end": production_rows["target_date"].max(),
        "n_production": len(production_rows),
        "return_rmse_pct": float(np.sqrt(mean_squared_error(test["target_return"], pred_return)) * 100),
    }

    # Testutfallen får ingå först efter utvärderingen, inom vald historiklängd.
    production_model = _fit_model(method, production_rows, feature_columns, best_params)

    joblib.dump(production_model, _model_path(method, horizon_weeks, macro_df is not None))
    return production_model, metrics


def load_model(method: str = "Random Forest", horizon_weeks: int = 1, macro_df: pd.DataFrame | None = None):
    path = _model_path(method, horizon_weeks, macro_df is not None)
    if not path.exists():
        return train_model(method=method, horizon_weeks=horizon_weeks, macro_df=macro_df)[0]
    return joblib.load(path)


def replace_latest_price_for_inference(df: pd.DataFrame, latest_price: float) -> pd.DataFrame:
    """Returnera en kopia där senaste veckopriset ersatts för en aktuell prognos.

    Originaldata och datum ändras inte. `pct_change` uppdateras om kolumnen finns,
    även om `build_features` räknar om den innan modellen används.
    """
    if df.empty or "date" not in df or "price" not in df:
        raise ValueError("Prognosdatan måste innehålla minst ett datum och pris.")
    price = float(latest_price)
    if not np.isfinite(price) or price <= 0:
        raise ValueError("Det senaste priset måste vara positivt och ändligt.")

    inference_df = df.copy()
    ordered_indices = pd.to_datetime(inference_df["date"]).sort_values().index
    latest_index = ordered_indices[-1]
    inference_df.at[latest_index, "price"] = price
    if "pct_change" in inference_df and len(ordered_indices) > 1:
        previous_price = float(inference_df.at[ordered_indices[-2], "price"])
        inference_df.at[latest_index, "pct_change"] = (price / previous_price - 1) * 100
    return inference_df


def predict_price(
    df: pd.DataFrame | None = None,
    model=None,
    method: str = "Random Forest",
    horizon_weeks: int = 1,
    macro_df: pd.DataFrame | None = None,
) -> float:
    """Förutspår priset `horizon_weeks` veckor efter senaste kända datapunkten."""
    if df is None:
        df = load_prices_df()
    if model is None:
        model = load_model(method, horizon_weeks, macro_df)

    feature_columns = _feature_columns(macro_df)
    feat = build_features(df, horizon_weeks, macro_df)
    latest = feat.iloc[[-1]]
    if latest[feature_columns].isna().any().any():
        raise ValueError("Prognosen kräver minst 53 veckopriser och kompletta, aktuella data för alla valda faktorer.")
    pred_return = model.predict(latest[feature_columns])[0]
    return float(_prices_from_returns(latest["price"].iloc[0], pred_return))


if __name__ == "__main__":
    for horizon_label, weeks in FORECAST_HORIZONS.items():
        print(f"=== Horisont: {horizon_label} ({weeks} veckor) ===")
        for method_name in METHODS:
            trained_model, m = train_model(method=method_name, horizon_weeks=weeks)
            price = predict_price(model=trained_model, horizon_weeks=weeks)
            val_rmse_str = f"{m['val_rmse']:>10,.0f}" if m["tuned"] else "     (ej tunad)"
            print(
                f"  {method_name:20s} val-RMSE {val_rmse_str}  "
                f"test-RMSE {m['rmse']:>12,.0f} (naiv {m['naive_rmse']:>12,.0f})  "
                f"params {m['best_params']}  prognos: {price:>12,.0f}"
            )
