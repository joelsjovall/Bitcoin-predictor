"""Makrodata och veckofaktorer för den separata Bitcoin-makromodellen."""
import numpy as np
import pandas as pd


MACRO_TICKERS = {"sp500": "^GSPC", "gold": "GC=F", "treasury_10y": "^TNX"}
MACRO_FEATURE_COLUMNS = [
    "sp500_return_12w",
    "sp500_return_52w",
    "gold_return_12w",
    "gold_return_52w",
    "treasury_10y_change_12w",
    "treasury_10y_level",
]


def fetch_macro_prices(start: str) -> pd.DataFrame:
    import yfinance as yf

    series = []
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    for column, ticker in MACRO_TICKERS.items():
        raw = yf.download(
            ticker, start=start, interval="1d", progress=False, auto_adjust=False,
        )
        if raw is None or raw.empty:
            raise RuntimeError(f"Inga makrodata för {ticker}. Försök hämta igen senare.")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        values = pd.to_numeric(raw["Close"], errors="coerce").rename(column)
        values.index = pd.to_datetime(values.index).tz_localize(None).normalize()
        values = values.loc[values.index < today].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            raise RuntimeError(f"Inga avslutade observationer för {ticker}.")
        series.append(values.loc[~values.index.duplicated(keep="last")])
    return pd.concat(series, axis=1).rename_axis("date").reset_index()


def build_macro_features(dates: pd.Series, macro_df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        pd.DataFrame({"date": pd.to_datetime(dates)})
        .sort_values("date")
        .reset_index(drop=True)
    )
    weekly["date"] = weekly["date"].dt.tz_localize(None).astype("datetime64[ns]")
    if not weekly["date"].diff().dropna().eq(pd.Timedelta(weeks=1)).all():
        raise ValueError("Makrofaktorer kräver sammanhängande veckodatum.")
    for column in MACRO_TICKERS:
        observations = macro_df[["date", column]].copy()
        observations["date"] = (
            pd.to_datetime(observations["date"])
            .dt.tz_localize(None)
            .astype("datetime64[ns]")
        )
        observations[column] = pd.to_numeric(observations[column], errors="coerce")
        observations = observations.replace([np.inf, -np.inf], np.nan).dropna()
        if column != "treasury_10y":
            observations = observations.loc[observations[column] > 0]
        observations = observations.sort_values("date").drop_duplicates("date", keep="last")
        weekly[column] = pd.merge_asof(
            weekly[["date"]],
            observations,
            on="date",
            direction="backward",
            tolerance=pd.Timedelta(days=7),
            allow_exact_matches=False,
        )[column]
    for asset in ("sp500", "gold"):
        for weeks in (12, 52):
            weekly[f"{asset}_return_{weeks}w"] = weekly[asset].pct_change(weeks, fill_method=None)
    weekly["treasury_10y_level"] = weekly["treasury_10y"]
    weekly["treasury_10y_change_12w"] = weekly["treasury_10y"].diff(12)
    return weekly[["date"] + MACRO_FEATURE_COLUMNS]
