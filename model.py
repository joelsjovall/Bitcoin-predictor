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

# Kandidater för hyperparameter-tuning, ett par per metod. Väljs mot valideringsdelen
# i train_model() (se där) – hålls medvetet få för att träningstiden inte ska explodera
# när fyra metoder x flera horisonter tränas om i appen.
PARAM_GRIDS = {
    "Linjär regression": [{}],
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
    return MODEL_DIR / f"{prefix}_{slug}_{horizon_weeks}w.pkl"


def build_features(df: pd.DataFrame, horizon_weeks: int = 1, macro_df: pd.DataFrame | None = None) -> pd.DataFrame:
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
    out["weeks_to_halving"] = (
        ESTIMATED_HALVING_INTERVAL_WEEKS - out["weeks_since_halving"]
    ).clip(lower=0).where(out["has_halving"].eq(1), 0)
    out["ret_lag_1"] = ret.shift(1)
    out["ret_lag_2"] = ret.shift(2)
    out["ret_lag_3"] = ret.shift(3)
    out["rolling_mean_return_4"] = ret.shift(1).rolling(4).mean()
    out["rolling_std_return_4"] = ret.shift(1).rolling(4).std()
    out["target_price"] = out["price"].shift(-horizon_weeks)
    out["target_date"] = out["date"].shift(-horizon_weeks)
    out["target_return"] = out["target_price"] / out["price"] - 1
    if macro_df is not None:
        out = out.merge(build_macro_features(out["date"], macro_df), on="date", how="left", validate="one_to_one")
    return out


MIN_EVAL_ROWS = 8  # minsta antal rader vi accepterar i en tränings-/valideringsdel


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


def _price_rmse(rows: pd.DataFrame, pred_return: np.ndarray) -> float:
    pred_price = rows["price"].values * (1 + pred_return)
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
    """Tränar `method` mot `horizon_weeks` med en kronologisk tränings-/validerings-/
    testdelning (standard 60/20/20).

    1. Ett par hyperparameter-kandidater (se PARAM_GRIDS) tränas på träningsdelen och
       utvärderas på valideringsdelen – den med lägst val-RMSE vinner.
    2. De vinnande hyperparametrarna tränas om på träning+validering och utvärderas en
       enda gång på den helt osedda testdelen – det är detta som rapporteras som
       modellens riktiga prestanda (mae/rmse/r2 i den returnerade metrics-dicten).
    3. Den slutliga modellen som faktiskt används för prognoser tränas om en sista gång
       med samma hyperparametrar men på *all* tillgänglig data (train+val+test), så att
       den verkliga prognosen får utnyttja så mycket historik som möjligt. Testdelens
       enda syfte är alltså att ge en ärlig uppskattning av hur bra den modellen är.

    För långa horisonter (t.ex. 5 år) räcker vår ~16-åriga historik inte till tre
    helt separata, läckagefria fönster (varje fönster behöver egen marginal på minst
    horizon_weeks rader). Då faller vi tillbaka på en enkel 80/20 train/test-delning
    med metodens standardhyperparametrar (ingen tuning) – `metrics["tuned"]` visar
    vilket som skedde.
    """
    if method not in METHODS:
        raise ValueError(f"Okänd metod: {method}. Välj bland {list(METHODS)}")
    if df is None:
        df = load_prices_df()
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size och test_size måste vara mellan 0 och 1 och summera till mindre än 1.")

    feature_columns = FEATURE_COLUMNS + (MACRO_FEATURE_COLUMNS if macro_df is not None else [])
    feat = build_features(df, horizon_weeks, macro_df).dropna(subset=feature_columns + ["target_return"])
    if len(feat) < 3:
        raise ValueError("För lite komplett historik för vald modell och horisont.")

    split = _three_way_split(feat, val_size, test_size, horizon_weeks)
    if split is not None:
        train, val, test = split
        best_params, best_val_rmse = None, np.inf
        for params in PARAM_GRIDS[method]:
            candidate = METHODS[method]()
            if params:
                candidate.set_params(**params)
            candidate.fit(train[feature_columns], train["target_return"])
            val_rmse = _price_rmse(val, candidate.predict(val[feature_columns]))
            if val_rmse < best_val_rmse:
                best_val_rmse, best_params = val_rmse, params
        train_for_eval = pd.concat([train, val])
        n_train, n_val, tuned = len(train), len(val), True
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
    eval_model = METHODS[method]()
    if best_params:
        eval_model.set_params(**best_params)
    eval_model.fit(train_for_eval[feature_columns], train_for_eval["target_return"])

    pred_return = eval_model.predict(test[feature_columns])
    pred_price = test["price"].values * (1 + pred_return)
    actual_price = test["target_price"].values
    naive_price = test["price"].values  # baseline: priset om N veckor = samma som nu

    metrics = {
        "method": method,
        "historical_margin_pct": historical_price_margin(actual_price, pred_price),
        "horizon_weeks": horizon_weeks,
        "tuned": tuned,
        "best_params": best_params,
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
        "production_train_start": feat["date"].min(),
        "production_train_end": feat["date"].max(),
        "production_target_end": feat["target_date"].max(),
        "n_production": len(feat),
        "return_rmse_pct": float(np.sqrt(mean_squared_error(test["target_return"], pred_return)) * 100),
    }

    # Produktionsmodell: samma hyperparametrar, tränad på all tillgänglig data.
    production_model = METHODS[method]()
    if best_params:
        production_model.set_params(**best_params)
    production_model.fit(feat[feature_columns], feat["target_return"])

    joblib.dump(production_model, _model_path(method, horizon_weeks, macro_df is not None))
    return production_model, metrics


def rolling_backtest(
    df: pd.DataFrame,
    method: str,
    horizon_weeks: int,
    holdout_start,
    macro_df: pd.DataFrame | None = None,
    step_weeks: int = 13,
    min_train_rows: int = 104,
):
    """Expanding-window backtest, with all scored outcomes before the final holdout.

    Refit at each origin. Tune only on outcomes known before that origin,
    purging training targets that overlap the inner validation period.
    Never reuse parameters chosen using later data or save these temporary models.
    """
    if method not in METHODS:
        raise ValueError("Okänd metod.")
    if not isinstance(step_weeks, int) or step_weeks < 1 or min_train_rows < MIN_EVAL_ROWS:
        raise ValueError("Ogiltigt teststeg eller för få träningsexempel.")
    columns = FEATURE_COLUMNS + (MACRO_FEATURE_COLUMNS if macro_df is not None else [])
    feat = build_features(df, horizon_weeks, macro_df).dropna(subset=columns + ["target_return"])
    eligible = feat.loc[feat["target_date"] < pd.Timestamp(holdout_start)]
    records = []
    next_origin = None
    for _, row in eligible.iterrows():
        origin = row["date"]
        if next_origin is not None and origin < next_origin:
            continue
        known = feat.loc[feat["target_date"] < origin]
        if len(known) < min_train_rows:
            continue
        params = {}
        val_count = max(MIN_EVAL_ROWS, int(len(known) * 0.2))
        validation = known.iloc[-val_count:]
        inner_train = known.loc[known["target_date"] < validation["date"].iloc[0]]
        tuned = len(inner_train) >= min_train_rows
        if tuned:
            best_score = np.inf
            for candidate_params in PARAM_GRIDS[method]:
                candidate = METHODS[method]().set_params(**candidate_params)
                candidate.fit(inner_train[columns], inner_train["target_return"])
                score = _price_rmse(validation, candidate.predict(validation[columns]))
                if score < best_score:
                    best_score, params = score, candidate_params
        estimator = METHODS[method]().set_params(**params)
        estimator.fit(known[columns], known["target_return"])
        inputs = feat.loc[[row.name], columns]
        predicted_return = float(estimator.predict(inputs)[0])
        records.append({
            "date": origin, "target_date": row["target_date"],
            "train_target_end": known["target_date"].max(), "n_train": len(known),
            "price": row["price"], "actual_price": row["target_price"],
            "predicted_price": row["price"] * (1 + predicted_return),
            "actual_return": row["target_return"], "predicted_return": predicted_return,
            "tuned": tuned,
        })
        next_origin = origin + pd.Timedelta(weeks=step_weeks)
    result = {"predictions": pd.DataFrame(records), "n_test": len(records),
              "step_weeks": step_weeks, "min_train_rows": min_train_rows}
    if not records:
        return result
    rows = result["predictions"]
    result.update({
        "rmse": float(np.sqrt(mean_squared_error(rows.actual_price, rows.predicted_price))),
        "naive_rmse": float(np.sqrt(mean_squared_error(rows.actual_price, rows.price))),
        "return_rmse_pct": float(np.sqrt(mean_squared_error(rows.actual_return, rows.predicted_return)) * 100),
        "historical_margin_pct": historical_price_margin(rows.actual_price, rows.predicted_price),
    })
    return result


def load_model(method: str = "Random Forest", horizon_weeks: int = 1, macro_df: pd.DataFrame | None = None):
    path = _model_path(method, horizon_weeks, macro_df is not None)
    if not path.exists():
        return train_model(method=method, horizon_weeks=horizon_weeks, macro_df=macro_df)[0]
    return joblib.load(path)


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

    feature_columns = FEATURE_COLUMNS + (MACRO_FEATURE_COLUMNS if macro_df is not None else [])
    feat = build_features(df, horizon_weeks, macro_df)
    latest = feat.iloc[[-1]]
    if latest[feature_columns].isna().any().any():
        raise ValueError("Prognosen kräver minst 53 veckopriser och kompletta, aktuella data för alla valda faktorer.")
    pred_return = model.predict(latest[feature_columns])[0]
    return float(latest["price"].iloc[0] * (1 + pred_return))


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
