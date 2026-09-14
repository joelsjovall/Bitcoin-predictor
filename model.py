"""AI-modellering: förutspår priset ett antal veckor framåt (regression)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from db import load_prices_df

MODEL_DIR = Path(__file__).parent

FEATURE_COLUMNS = [
    "pct_change",
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "rolling_mean_return_4",
    "rolling_std_return_4",
]

# Alla metoder tränas på samma features/target, se build_features(). Features skalas
# (StandardScaler) eftersom SVR och linjär regression är känsliga för det, medan det är
# harmlöst för de trädbaserade metoderna.
METHODS = {
    "Linjär regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "SVR": lambda: make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.01)),
    "Random Forest": lambda: make_pipeline(
        StandardScaler(), RandomForestRegressor(n_estimators=300, random_state=42)
    ),
    "XGBoost": lambda: make_pipeline(
        StandardScaler(), XGBRegressor(n_estimators=300, random_state=42)
    ),
}

FORECAST_HORIZONS = {
    "1 månad": 4,
    "3 månader": 13,
    "6 månader": 26,
    "1 år": 52,
    "3 år": 156,
    "5 år": 260,
}


def _model_path(method: str, horizon_weeks: int) -> Path:
    slug = method.lower().replace(" ", "_").replace("ä", "a").replace("ö", "o")
    return MODEL_DIR / f"model_{slug}_{horizon_weeks}w.pkl"


def build_features(df: pd.DataFrame, horizon_weeks: int = 1) -> pd.DataFrame:
    """Bygger features och ett mål `horizon_weeks` veckor framåt.

    df måste innehålla kolumnerna date, price, pct_change och vara sorterad äldst
    först. Bitcoin-priset har vuxit exponentiellt (från ca 0,1 till 80 000+ USD), så
    absoluta prisnivåer generaliserar dåligt mellan tränings- och testperiod. Därför
    byggs features och mål som relativa förändringar (avkastning) istället.

    Modellen tränas direkt mot horisonten (t.ex. "priset om 52 veckor"), inte genom
    att kedja ihop upprepade enveckasprognoser – det senare får fel att ackumuleras
    exponentiellt över långa horisonter.
    """
    out = df.copy().sort_values("date").reset_index(drop=True)
    ret = out["price"].pct_change()
    out["ret_lag_1"] = ret.shift(1)
    out["ret_lag_2"] = ret.shift(2)
    out["ret_lag_3"] = ret.shift(3)
    out["rolling_mean_return_4"] = ret.shift(1).rolling(4).mean()
    out["rolling_std_return_4"] = ret.shift(1).rolling(4).std()
    out["target_price"] = out["price"].shift(-horizon_weeks)
    out["target_return"] = out["target_price"] / out["price"] - 1
    return out


def train_model(
    df: pd.DataFrame | None = None,
    method: str = "Random Forest",
    horizon_weeks: int = 1,
    test_size: float = 0.2,
):
    if method not in METHODS:
        raise ValueError(f"Okänd metod: {method}. Välj bland {list(METHODS)}")
    if df is None:
        df = load_prices_df()

    feat = build_features(df, horizon_weeks).dropna(subset=FEATURE_COLUMNS + ["target_return"])

    split_idx = int(len(feat) * (1 - test_size))
    train, test = feat.iloc[:split_idx], feat.iloc[split_idx:]

    X_train, y_train = train[FEATURE_COLUMNS], train["target_return"]
    X_test, y_test = test[FEATURE_COLUMNS], test["target_return"]

    model = METHODS[method]()
    model.fit(X_train, y_train)

    pred_return = model.predict(X_test)
    pred_price = test["price"].values * (1 + pred_return)
    actual_price = test["target_price"].values
    naive_price = test["price"].values  # baseline: priset om N veckor = samma som nu

    metrics = {
        "method": method,
        "horizon_weeks": horizon_weeks,
        "mae": mean_absolute_error(actual_price, pred_price),
        "rmse": np.sqrt(mean_squared_error(actual_price, pred_price)),
        "r2": r2_score(actual_price, pred_price),
        "naive_mae": mean_absolute_error(actual_price, naive_price),
        "naive_rmse": np.sqrt(mean_squared_error(actual_price, naive_price)),
        "n_train": len(train),
        "n_test": len(test),
    }

    joblib.dump(model, _model_path(method, horizon_weeks))
    return model, metrics


def load_model(method: str = "Random Forest", horizon_weeks: int = 1):
    path = _model_path(method, horizon_weeks)
    if not path.exists():
        return train_model(method=method, horizon_weeks=horizon_weeks)[0]
    return joblib.load(path)


def predict_price(
    df: pd.DataFrame | None = None,
    model=None,
    method: str = "Random Forest",
    horizon_weeks: int = 1,
) -> float:
    """Förutspår priset `horizon_weeks` veckor efter senaste kända datapunkten."""
    if df is None:
        df = load_prices_df()
    if model is None:
        model = load_model(method, horizon_weeks)

    feat = build_features(df, horizon_weeks)
    latest = feat.iloc[[-1]]
    pred_return = model.predict(latest[FEATURE_COLUMNS])[0]
    return float(latest["price"].iloc[0] * (1 + pred_return))


if __name__ == "__main__":
    for horizon_label, weeks in FORECAST_HORIZONS.items():
        print(f"=== Horisont: {horizon_label} ({weeks} veckor) ===")
        for method_name in METHODS:
            trained_model, m = train_model(method=method_name, horizon_weeks=weeks)
            price = predict_price(model=trained_model, horizon_weeks=weeks)
            print(
                f"  {method_name:20s} RMSE {m['rmse']:>12,.0f} "
                f"(naiv {m['naive_rmse']:>12,.0f})  prognos: {price:>12,.0f}"
            )
