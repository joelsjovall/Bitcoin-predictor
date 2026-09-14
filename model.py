"""AI-modellering: förutspår nästa veckas stängningskurs (regression)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from db import load_prices_df

MODEL_PATH = Path(__file__).parent / "model.pkl"

FEATURE_COLUMNS = [
    "pct_change",
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "rolling_mean_return_4",
    "rolling_std_return_4",
    "volume_ratio_4",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bygger features utifrån historiken. df måste vara sorterad på datum, äldst först.

    Bitcoin-priset har vuxit exponentiellt (från ca 0,1 till 80 000+ USD), så absoluta
    prisnivåer generaliserar dåligt mellan tränings- och testperiod. Därför byggs features
    och target som relativa förändringar (avkastning) istället för absoluta prisnivåer.
    """
    out = df.copy().sort_values("date").reset_index(drop=True)
    ret = out["close"].pct_change()
    out["ret_lag_1"] = ret.shift(1)
    out["ret_lag_2"] = ret.shift(2)
    out["ret_lag_3"] = ret.shift(3)
    out["rolling_mean_return_4"] = ret.shift(1).rolling(4).mean()
    out["rolling_std_return_4"] = ret.shift(1).rolling(4).std()
    out["volume_ratio_4"] = out["volume"] / out["volume"].shift(1).rolling(4).mean()
    out["target_return"] = out["close"].shift(-1) / out["close"] - 1
    out["target_next_close"] = out["close"].shift(-1)
    return out


def train_model(df: pd.DataFrame | None = None, test_size: float = 0.2):
    if df is None:
        df = load_prices_df()

    feat = build_features(df).dropna(subset=FEATURE_COLUMNS + ["target_return"])

    split_idx = int(len(feat) * (1 - test_size))
    train, test = feat.iloc[:split_idx], feat.iloc[split_idx:]

    X_train, y_train = train[FEATURE_COLUMNS], train["target_return"]
    X_test, y_test = test[FEATURE_COLUMNS], test["target_return"]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    pred_return = model.predict(X_test)
    pred_price = test["close"].values * (1 + pred_return)
    actual_price = test["target_next_close"].values
    naive_price = test["close"].values  # baseline: nästa vecka = samma som denna vecka

    metrics = {
        "mae": mean_absolute_error(actual_price, pred_price),
        "rmse": np.sqrt(mean_squared_error(actual_price, pred_price)),
        "r2": r2_score(actual_price, pred_price),
        "naive_mae": mean_absolute_error(actual_price, naive_price),
        "naive_rmse": np.sqrt(mean_squared_error(actual_price, naive_price)),
        "n_train": len(train),
        "n_test": len(test),
    }

    joblib.dump(model, MODEL_PATH)
    return model, metrics


def load_model():
    if not MODEL_PATH.exists():
        return train_model()[0]
    return joblib.load(MODEL_PATH)


def predict_next_close(df: pd.DataFrame | None = None, model=None) -> float:
    if df is None:
        df = load_prices_df()
    if model is None:
        model = load_model()

    feat = build_features(df)
    latest = feat.iloc[[-1]]
    pred_return = model.predict(latest[FEATURE_COLUMNS])[0]
    return float(latest["close"].iloc[0] * (1 + pred_return))


if __name__ == "__main__":
    trained_model, m = train_model()
    print("Modell tränad. Utvärdering på testperioden:")
    print(f"  MAE:  {m['mae']:.2f}  (naiv baseline: {m['naive_mae']:.2f})")
    print(f"  RMSE: {m['rmse']:.2f}  (naiv baseline: {m['naive_rmse']:.2f})")
    print(f"  R2:   {m['r2']:.3f}")
    print(f"  Tränad på {m['n_train']} veckor, testad på {m['n_test']} veckor")
    print(f"Prognos nästa vecka: {predict_next_close(model=trained_model):.1f}")
