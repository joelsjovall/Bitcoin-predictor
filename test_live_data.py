import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from db import fetch_live_prices, ingest_csv, ingest_live, load_prices_df


class LiveDataTests(unittest.TestCase):
    def test_daily_prices_become_sunday_prices(self):
        raw = pd.DataFrame(
            {"Close": range(100, 114)},
            index=pd.date_range("2025-01-06", periods=14, name="Date"),
        )
        with patch("yfinance.download", return_value=raw):
            result = fetch_live_prices()
        self.assertEqual(result.date.dt.strftime("%Y-%m-%d").tolist(), ["2025-01-12", "2025-01-19"])
        self.assertEqual(result.price.tolist(), [106, 113])

    def test_empty_download_is_error(self):
        with patch("yfinance.download", return_value=pd.DataFrame()):
            with self.assertRaises(RuntimeError):
                fetch_live_prices()

    def test_database_update_and_invalid_dates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.db"
            ingest_csv(db_path=path)
            before = load_prices_df(path)
            latest = before.date.max()
            update = pd.DataFrame({
                "date": [latest, latest + pd.Timedelta(weeks=1)],
                "price": [80000.0, 81000.0],
                "pct_change": [0.0, 1.25],
            })
            with patch("db.fetch_live_prices", return_value=update):
                self.assertEqual(ingest_live(db_path=path), 2)
            after = load_prices_df(path)
            self.assertEqual(len(after), len(before) + 1)
            self.assertEqual(after.price.iloc[-2:].tolist(), [80000.0, 81000.0])
            pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-2])
            update["date"] += pd.Timedelta(days=1)
            with patch("db.fetch_live_prices", return_value=update):
                with self.assertRaises(ValueError):
                    ingest_live(db_path=path)
            pd.testing.assert_frame_equal(after, load_prices_df(path))


if __name__ == "__main__":
    unittest.main()
