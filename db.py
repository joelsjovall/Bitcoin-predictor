"""Backend: läser in CSV-datan och lagrar den i en SQLite-databas."""
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
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
    """Skapar pristabellen med datum, pris, prisförändring och volym.

    Migrerar en äldre tabell som saknar PRIMARY KEY på date (behövs för
    ON CONFLICT i ingest_live()) genom att bygga om den och kopiera över datan.
    """
    with closing(sqlite3.connect(db_path)) as conn:
        schema_sql = """
            CREATE TABLE prices (
                date TEXT PRIMARY KEY,
                price REAL,
                pct_change REAL,
                volume REAL
            )
            """
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'prices'"
        ).fetchone()

        if not table_exists:
            conn.execute(schema_sql)

        table_info = conn.execute("PRAGMA table_info(prices)").fetchall()
        has_date_primary_key = any(row[1] == "date" and row[5] for row in table_info)
        if table_info and not has_date_primary_key:
            conn.execute("ALTER TABLE prices RENAME TO prices_old")
            conn.execute(schema_sql)

            old_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(prices_old)")
            }
            copy_columns = [
                column
                for column in ["date", "price", "pct_change", "volume"]
                if column in old_columns
            ]
            if copy_columns:
                columns_sql = ", ".join(copy_columns)
                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO prices ({columns_sql})
                    SELECT {columns_sql}
                    FROM prices_old
                    WHERE date IS NOT NULL
                    """
                )
            conn.execute("DROP TABLE prices_old")
        current_columns = {row[1] for row in conn.execute("PRAGMA table_info(prices)")}
        if current_columns and "volume" not in current_columns:
            conn.execute("ALTER TABLE prices ADD COLUMN volume REAL")

        conn.commit()


def ingest_csv(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH) -> int:
    """Läser CSV:n och skriver (eller ersätter) datan i databasen. Returnerar antal rader.

    Tömmer tabellen och fyller på den igen (istället för to_sql-replace) så att
    schemat, inklusive PRIMARY KEY på date, bevaras – annat skulle ON CONFLICT i
    ingest_live() sluta fungera.
    """
    df = load_csv(csv_path)
    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("DELETE FROM prices")
        df.assign(volume=None, date=df["date"].dt.strftime("%Y-%m-%d")).to_sql(
            "prices", conn, if_exists="append", index=False
        )
        conn.commit()
    return len(df)


def fetch_live_prices(start: str = "2011-09-15", interval: str = "1wk") -> pd.DataFrame:
    """Hämtar BTC-USD-priser från Yahoo Finance från och med `start` till idag
    (samma schema som CSV-datan)."""
    import yfinance as yf

    if interval != "1wk":
        raise ValueError("Modellen kräver veckodata (interval='1wk').")
    # Dagliga stängningspriser ger gemensamma söndagsdatum med CSV-filen.
    raw = yf.download(
        "BTC-USD", start=start, interval="1d", progress=False, auto_adjust=False
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returnerade inga priser. Kontrollera anslutningen och försök igen.")

    raw = raw.reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw["Volume"] = pd.to_numeric(raw["Volume"], errors="coerce") if "Volume" in raw else np.nan
    raw["weekly_volume"] = raw["Volume"].rolling(7, min_periods=7).sum()
    df = raw.rename(
        columns={"Date": "date", "Close": "price", "weekly_volume": "volume"}
    )[["date", "price", "volume"]]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    df = df.loc[df["date"] < today].dropna(subset=["price"])
    # Ta endast faktiska söndagsstängningar; fyll inte luckor med äldre priser.
    df = (
        df.loc[df["date"].dt.dayofweek == 6]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if df.empty:
        raise RuntimeError("Yahoo Finance returnerade inga avslutade söndagspriser.")
    df["pct_change"] = df["price"].pct_change(fill_method=None) * 100
    return df


def ingest_live(db_path: Path = DB_PATH, start: str = "2011-09-15", interval: str = "1wk") -> int:
    """Hämtar data från Yahoo Finance (från `start` till idag) och skriver in den i
    databasen (uppdaterar matchande datum och behåller äldre CSV-historik).
    Returnerar antal rader som hämtades."""
    df = fetch_live_prices(start=start, interval=interval)
    if df.empty:
        return 0
    if (df["date"].dt.dayofweek != 6).any():
        raise ValueError("Live-data måste vara söndagsdaterad veckodata (samma vecko-cykel som CSV-historiken).")

    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        rows = df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_dict("records")
        for row in rows:
            row.setdefault("volume", None)
        conn.executemany(
            """
            INSERT INTO prices (date, price, pct_change, volume)
            VALUES (:date, :price, :pct_change, :volume)
            ON CONFLICT(date) DO UPDATE SET
                price=excluded.price,
                pct_change=excluded.pct_change,
                volume=excluded.volume
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def load_prices_df(db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        ingest_csv(db_path=db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        return pd.read_sql("SELECT * FROM prices ORDER BY date", conn, parse_dates=["date"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Läs in Bitcoin-priser i databasen.")
    parser.add_argument("--live", action="store_true", help="Hämta senaste avslutade veckopriser från Yahoo Finance")
    args = parser.parse_args()
    n = ingest_live() if args.live else ingest_csv()
    print(f"Lade in {n} rader i {DB_PATH}")
