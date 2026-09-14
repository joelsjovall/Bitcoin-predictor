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
    """Läser CSV:n och skriver (eller ersätter) datan i databasen. Returnerar antal rader."""
    df = load_csv(csv_path)
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_sql(
        "prices", conn, if_exists="replace", index=False
    )
    conn.commit()
    conn.close()
    return len(df)


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
