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
HALVINGS = pd.to_datetime(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"])

FEATURE_COLUMNS = [
    "return_1w",
    "return_4w",
    "return_12w",
    "return_52w",
    "volatility_12w",
    "volatility_52w",
    "distance_from_ath",
    "weeks_since_halving",
    "has_halving",
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
    "2 år": 104,
    "3 år": 156,
    "5 år": 260,
}


def _model_path(method: str, horizon_weeks: int) -> Path:
    slug = method.lower().replace(" ", "_").replace("ä", "a").replace("ö", "o")
    return MODEL_DIR / f"model_v2_{slug}_{horizon_weeks}w.pkl"


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
    out["ret_lag_1"] = ret.shift(1)
    out["ret_lag_2"] = ret.shift(2)
    out["ret_lag_3"] = ret.shift(3)
    out["rolling_mean_return_4"] = ret.shift(1).rolling(4).mean()
    out["rolling_std_return_4"] = ret.shift(1).rolling(4).std()
    out["target_price"] = out["price"].shift(-horizon_weeks)
    out["target_date"] = out["date"].shift(-horizon_weeks)
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

    if not 0 < test_size < 1:
        raise ValueError("test_size måste vara mellan 0 och 1.")
    split_idx = int(len(feat) * (1 - test_size))
    train, test = feat.iloc[:split_idx], feat.iloc[split_idx:]
    if train.empty or len(test) < 2:
        raise ValueError("För lite historik för vald horisont och testperiod.")
    train = train.loc[train["target_date"] < test["date"].iloc[0]]
    if train.empty:
        raise ValueError("För lite historik för att skilja träningsmål från testperioden.")

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
    if latest[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Prognosen kräver minst 53 veckopriser.")
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
