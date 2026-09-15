from pathlib import Path
import sqlite3

import pandas as pd
import pytest

import db as db_module
from db import ingest_csv, ingest_live, load_csv, load_prices_df


def write_sample_csv(path: Path) -> None:
    path.write_text(
        '"Datum","Senaste","+/- %"\n'
        '"2024-01-14","78.543,5","-1,60%"\n'
        '"2024-01-07","1.234,0","2,04%"\n'
        '"2024-01-21","80.000,0","0,50%"\n',
        encoding="utf-8",
    )


def test_load_csv_parses_swedish_numbers_and_sorts_by_date(tmp_path):
    csv_path = tmp_path / "bitcoin.csv"
    write_sample_csv(csv_path)

    df = load_csv(csv_path)

    assert list(df.columns) == ["date", "price", "pct_change"]
    assert df["date"].tolist() == list(pd.to_datetime(["2024-01-07", "2024-01-14", "2024-01-21"]))
    assert df["price"].tolist() == pytest.approx([1234.0, 78543.5, 80000.0])
    assert df["pct_change"].tolist() == pytest.approx([2.04, -1.60, 0.50])


def test_ingest_csv_writes_to_sqlite_and_loads_rows_back(tmp_path):
    csv_path = tmp_path / "bitcoin.csv"
    db_path = tmp_path / "bitcoin.db"
    write_sample_csv(csv_path)

    row_count = ingest_csv(csv_path=csv_path, db_path=db_path)
    df = load_prices_df(db_path=db_path)

    assert row_count == 3
    assert db_path.exists()
    assert len(df) == 3
    assert {"date", "price", "pct_change"}.issubset(df.columns)
    assert df.iloc[0]["date"] == pd.Timestamp("2024-01-07")
    assert df.iloc[-1]["price"] == pytest.approx(80000.0)


def test_ingest_live_upserts_rows_without_network(tmp_path, monkeypatch):
    csv_path = tmp_path / "bitcoin.csv"
    db_path = tmp_path / "bitcoin.db"
    write_sample_csv(csv_path)
    ingest_csv(csv_path=csv_path, db_path=db_path)

    live_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-21", "2024-01-28"]),
            "price": [81_500.0, 83_000.0],
            "pct_change": [1.875, 1.840],
        }
    )
    monkeypatch.setattr(db_module, "fetch_live_prices", lambda start="2021-09-15", interval="1wk": live_df)

    row_count = ingest_live(db_path=db_path)
    df = load_prices_df(db_path=db_path)

    assert row_count == 2
    assert len(df) == 4
    assert {"date", "price", "pct_change"}.issubset(df.columns)

    updated_row = df.loc[df["date"] == pd.Timestamp("2024-01-21")].iloc[0]
    assert updated_row["price"] == pytest.approx(81_500.0)

    inserted_row = df.loc[df["date"] == pd.Timestamp("2024-01-28")].iloc[0]
    assert inserted_row["price"] == pytest.approx(83_000.0)


def test_ingest_live_migrates_old_table_without_date_primary_key(tmp_path, monkeypatch):
    db_path = tmp_path / "bitcoin.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE prices (date TEXT, price REAL, pct_change REAL)")
    conn.execute(
        "INSERT INTO prices (date, price, pct_change) VALUES ('2024-01-21', 80000.0, 0.50)"
    )
    conn.commit()
    conn.close()

    live_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-21"]),
            "price": [81_500.0],
            "pct_change": [1.875],
        }
    )
    monkeypatch.setattr(db_module, "fetch_live_prices", lambda start="2021-09-15", interval="1wk": live_df)

    ingest_live(db_path=db_path)

    df = load_prices_df(db_path=db_path)
    assert len(df) == 1
    assert df.iloc[0]["price"] == pytest.approx(81_500.0)

    conn = sqlite3.connect(db_path)
    table_info = conn.execute("PRAGMA table_info(prices)").fetchall()
    conn.close()
    assert any(row[1] == "date" and row[5] for row in table_info)
