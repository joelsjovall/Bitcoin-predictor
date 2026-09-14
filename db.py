"""Backend: läser in CSV-datan och lagrar den i en SQLite-databas."""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "bitcoin.db"
CSV_PATH = Path(__file__).parent / "Bitcoin Empirisk data.csv"


def _parse_number(value: str) -> float:
    """'78.543,5' -> 78543.5"""
    return float(value.replace(".", "").replace(",", "."))


def _parse_percent(value: str) -> float:
    """'-1,60%' -> -1.60"""
    return _parse_number(value.replace("%", ""))


def load_csv(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.rename(
        columns={
            "Datum": "date",
            "Senaste": "price",
            "+/- %": "pct_change",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = df["price"].apply(_parse_number)
    df["pct_change"] = df["pct_change"].apply(_parse_percent)
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "price", "pct_change"]]


def init_db(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            date TEXT PRIMARY KEY,
            price REAL,
            pct_change REAL
        )
        """
    )
    conn.commit()
    conn.close()


def ingest_csv(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH) -> int:
    """Läser CSV:n och skriver (eller ersätter) datan i databasen. Returnerar antal rader.

    Tömmer tabellen och fyller på den igen (istället för to_sql-replace) så att
    schemat, inklusive PRIMARY KEY på date, bevaras – annat skulle ON CONFLICT i
    ingest_live() sluta fungera.
    """
    df = load_csv(csv_path)
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM prices")
    df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_sql(
        "prices", conn, if_exists="append", index=False
    )
    conn.commit()
    conn.close()
    return len(df)


def fetch_live_prices(period: str = "2y", interval: str = "1wk") -> pd.DataFrame:
    """Hämtar senaste BTC-USD-priser från Yahoo Finance (samma schema som CSV-datan)."""
    import yfinance as yf

    raw = yf.download("BTC-USD", period=period, interval=interval, progress=False, auto_adjust=False)
    if raw.empty:
        return pd.DataFrame(columns=["date", "price", "pct_change"])

    raw = raw.reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.rename(columns={"Date": "date", "Close": "price"})[["date", "price"]]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    df["pct_change"] = df["price"].pct_change() * 100
    return df.dropna(subset=["price"])


def ingest_live(db_path: Path = DB_PATH, period: str = "2y", interval: str = "1wk") -> int:
    """Hämtar aktuell data från Yahoo Finance och skriver in den i databasen (kompletterar/uppdaterar,
    skriver inte över den historiska CSV-datan). Returnerar antal rader som hämtades."""
    df = fetch_live_prices(period=period, interval=interval)
    if df.empty:
        return 0

    init_db(db_path)
    conn = sqlite3.connect(db_path)
    rows = df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_dict("records")
    conn.executemany(
        """
        INSERT INTO prices (date, price, pct_change)
        VALUES (:date, :price, :pct_change)
        ON CONFLICT(date) DO UPDATE SET
            price=excluded.price, pct_change=excluded.pct_change
        """,
        rows,
    )
    conn.commit()
    conn.close()
    return len(rows)


def load_prices_df(db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        ingest_csv(db_path=db_path)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM prices ORDER BY date", conn, parse_dates=["date"])
    conn.close()
    return df


if __name__ == "__main__":
    n = ingest_csv()
    print(f"Lade in {n} rader i {DB_PATH}")
