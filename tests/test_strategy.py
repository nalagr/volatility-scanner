from datetime import datetime
import math
import os
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add src to sys.path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import analyze
import backtest
import database
import fetch
import volatility


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.orig_db_path = database.DB_PATH
        self.test_db = Path(__file__).parent / "test_trading.db"
        database.DB_PATH = self.test_db
        database.create_tables()

    def tearDown(self):
        database.DB_PATH = self.orig_db_path
        if self.test_db.exists():
            try:
                self.test_db.unlink()
            except Exception:
                pass

    def test_wal_and_pragmas(self):
        conn = database.create_connection()
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        busy_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        conn.close()
        self.assertEqual(journal_mode.lower(), "wal")
        self.assertEqual(busy_timeout, 5000)
        self.assertEqual(foreign_keys, 1)

    def test_tables_created(self):
        conn = database.create_connection()
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        self.assertIn("stocks", tables)
        self.assertIn("stock_prices", tables)
        self.assertIn("volatility_snapshots", tables)
        self.assertIn("positions", tables)

    def test_position_crud_and_lifecycle(self):
        # 1. Insert position
        pos_id = database.insert_position(
            ticker="AAPL",
            entry_date="2026-10-27",
            long_strike=300.0,
            long_expiry="2027-01-15",
            long_cost=45.0,
            notes="Initial Phase 1"
        )
        self.assertGreater(pos_id, 0)

        # 2. Get open positions
        open_pos = database.get_open_positions()
        self.assertEqual(len(open_pos), 1)
        self.assertEqual(open_pos[0]["ticker"], "AAPL")
        self.assertEqual(open_pos[0]["status"], "PHASE1_OPEN")
        self.assertEqual(open_pos[0]["net_cost_basis"], 45.0)

        # 3. Hedge position into Phase 2
        success = database.update_position_hedge(
            position_id=pos_id,
            short_strike=360.0,
            short_expiry="2026-11-20",
            short_credit=3.0,
            notes="OTM Call sold"
        )
        self.assertTrue(success)

        hedged_pos = database.get_positions(status="PHASE2_HEDGED")
        self.assertEqual(len(hedged_pos), 1)
        self.assertEqual(hedged_pos[0]["net_cost_basis"], 42.0)
        self.assertIn("OTM Call sold", hedged_pos[0]["notes"])

        # 4. Roll hedge forward with cumulative basis reduction
        rolled = database.roll_position_hedge(
            position_id=pos_id,
            new_short_strike=370.0,
            new_short_expiry="2026-12-18",
            new_short_credit=2.50,
            notes="Month 2 roll"
        )
        self.assertTrue(rolled)
        rolled_pos = database.get_positions(status="PHASE2_HEDGED")
        self.assertEqual(rolled_pos[0]["net_cost_basis"], 39.50)  # 42.0 - 2.50

        # 5. Unhedge when short call expires worthless
        unhedged = database.unhedge_position(pos_id, notes="Expired worthless")
        self.assertTrue(unhedged)
        open_again = database.get_open_positions()
        self.assertEqual(len(open_again), 1)
        self.assertEqual(open_again[0]["status"], "PHASE1_OPEN")
        self.assertEqual(open_again[0]["net_cost_basis"], 39.50)  # Basis preserved!

        # 6. Close position
        closed = database.close_position(pos_id, notes="Exited position")
        self.assertTrue(closed)
        closed_pos = database.get_positions(status="CLOSED")
        self.assertEqual(len(closed_pos), 1)
        self.assertEqual(closed_pos[0]["status"], "CLOSED")

    def test_get_latest_price_date(self):
        database.insert_stock("TEST")
        self.assertIsNone(database.get_latest_price_date("TEST"))
        database.insert_price("TEST", "2026-10-20", 100, 105, 99, 104, 1000)
        database.insert_price("TEST", "2026-10-25", 104, 108, 103, 107, 1200)
        latest = database.get_latest_price_date("TEST")
        self.assertEqual(latest, "2026-10-25")

    def test_snapshot_iv_rank_update(self):
        database.insert_volatility("TEST", 0.20, 0.25, 0.0)
        zero_snaps = database.get_zero_rank_snapshots()
        self.assertEqual(len(zero_snaps), 1)
        snap_id = zero_snaps[0]["id"]

        updated = database.update_snapshot_iv_rank(snap_id, 45.5)
        self.assertTrue(updated)
        remaining_zero = database.get_zero_rank_snapshots()
        self.assertEqual(len(remaining_zero), 0)


class TestVolatility(unittest.TestCase):
    def setUp(self):
        self.orig_db_path = database.DB_PATH
        self.test_db = Path(__file__).parent / "test_vol.db"
        database.DB_PATH = self.test_db
        database.create_tables()

    def tearDown(self):
        database.DB_PATH = self.orig_db_path
        if self.test_db.exists():
            try:
                self.test_db.unlink()
            except Exception:
                pass

    def test_calc_20day_hv(self):
        self.assertIsNone(volatility.calc_20day_hv("TEST"))

        prices = [
            ("TEST", f"2026-09-{i:02d}", 100 + i, 101 + i, 99 + i, 100 + i, 1000)
            for i in range(1, 23)
        ]
        database.insert_prices_batch(prices)

        hv = volatility.calc_20day_hv("TEST")
        self.assertIsNotNone(hv)
        self.assertGreater(hv, 0.0)

    def test_calc_iv_rank_proxy(self):
        prices = [
            ("TEST", f"2026-{1 + (i//30):02d}-{1 + (i%28):02d}", 100 + math.sin(i)*10, 105, 95, 100 + math.sin(i)*10, 1000)
            for i in range(70)
        ]
        database.insert_prices_batch(prices)

        iv_proxy = volatility.calc_iv_rank_proxy("TEST", 0.30)
        self.assertIsNotNone(iv_proxy)
        self.assertGreaterEqual(iv_proxy, 0.0)
        self.assertLessEqual(iv_proxy, 100.0)

    def test_black_scholes_call_greeks(self):
        # Deep ITM call: spot=300, strike=260, 75 DTE, IV=25%
        itm = volatility.black_scholes_call(300.0, 260.0, 75, 0.25)
        self.assertGreaterEqual(itm["price"], 40.0)  # Intrinsic floor (300-260=40)
        self.assertGreaterEqual(itm["delta"], 0.80)   # High delta
        self.assertLess(itm["theta"], 0.0)           # Time decay is negative
        self.assertGreater(itm["vega"], 0.0)          # Vega is positive

        # OTM call: spot=300, strike=320, 25 DTE, IV=25%
        otm = volatility.black_scholes_call(300.0, 320.0, 25, 0.25)
        self.assertGreater(otm["price"], 0.0)
        self.assertLess(otm["delta"], 0.40)          # OTM call delta < 0.40
        self.assertGreater(otm["delta"], 0.10)

        # Zero DTE edge case
        zero_dte = volatility.black_scholes_call(300.0, 280.0, 0, 0.25)
        self.assertEqual(zero_dte["price"], 20.0)
        self.assertEqual(zero_dte["delta"], 1.0)

    def test_calc_iv_rank_empirical(self):
        # Insert 25 historical snapshots
        for iv in [0.20 + (i * 0.01) for i in range(25)]:
            database.insert_volatility("TEST_EMP", 0.15, iv, 50.0)

        # Current IV = 0.20 -> rank should be 0.0%
        rank_low = volatility.calc_iv_rank("TEST_EMP", 0.20)
        self.assertAlmostEqual(rank_low, 0.0, delta=1.0)

        # Current IV = 0.44 -> rank should be 100.0%
        rank_high = volatility.calc_iv_rank("TEST_EMP", 0.44)
        self.assertAlmostEqual(rank_high, 100.0, delta=1.0)


class TestAnalyze(unittest.TestCase):
    def setUp(self):
        self.orig_db_path = database.DB_PATH
        self.test_db = Path(__file__).parent / "test_analyze.db"
        database.DB_PATH = self.test_db
        database.create_tables()

    def tearDown(self):
        database.DB_PATH = self.orig_db_path
        if self.test_db.exists():
            try:
                self.test_db.unlink()
            except Exception:
                pass

    def test_position_tracking_commands(self):
        # 1. Add
        analyze.add_position_cmd("AAPL", 300.0, "2027-01-15", 40.0, "Test Position")
        positions = database.get_open_positions()
        self.assertEqual(len(positions), 1)
        pos_id = positions[0]["id"]

        # 2. Hedge
        analyze.hedge_position_cmd(pos_id, 350.0, "2026-11-20", 3.0, "Hedge sold")
        hedged = database.get_positions(status="PHASE2_HEDGED")
        self.assertEqual(len(hedged), 1)
        self.assertEqual(hedged[0]["net_cost_basis"], 37.0)

        # 3. Roll
        analyze.roll_position_cmd(pos_id, 360.0, "2026-12-18", 2.0, "Second month hedge")
        rolled = database.get_positions(status="PHASE2_HEDGED")
        self.assertEqual(rolled[0]["net_cost_basis"], 35.0)

        # 4. Unhedge
        analyze.unhedge_position_cmd(pos_id, "Hedge expired")
        unhedged = database.get_open_positions()
        self.assertEqual(len(unhedged), 1)
        self.assertEqual(unhedged[0]["net_cost_basis"], 35.0)

        # 5. Close
        analyze.close_position_cmd(pos_id, "Profit taken")
        closed = database.get_positions(status="CLOSED")
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["status"], "CLOSED")

    def test_check_position_alerts_crash_resilience(self):
        # Verify check_position_alerts handles None spot price and missing options without crashing
        analyze.add_position_cmd("MOCK", 100.0, "2027-01-15", 15.0, "Alert Test")

        # Mock yfinance to simulate missing network quotes / closed market
        with patch("analyze.yf.Ticker") as mock_ticker, \
             patch("analyze.calc_20day_hv", return_value=0.20), \
             patch("analyze.atm_iv", return_value=0.45), \
             patch("analyze.calc_iv_rank", return_value=85.0), \
             patch("analyze.get_bloated_otm_call", return_value=None):

            mock_stock = MagicMock()
            mock_stock.fast_info = {}  # Empty fast_info (spot_price = None)
            mock_ticker.return_value = mock_stock

            # Must execute without throwing any exception!
            try:
                analyze.check_position_alerts()
            except Exception as e:
                self.fail(f"check_position_alerts crashed unexpectedly: {e}")


class TestBacktest(unittest.TestCase):
    def setUp(self):
        self.orig_db_path = database.DB_PATH
        self.test_db = Path(__file__).parent / "test_bt.db"
        database.DB_PATH = self.test_db
        database.create_tables()

    def tearDown(self):
        database.DB_PATH = self.orig_db_path
        if self.test_db.exists():
            try:
                self.test_db.unlink()
            except Exception:
                pass

    def test_run_backtest(self):
        # Insert 120 days of simulated price data
        prices = [
            ("SIM", f"2026-{1 + (i//28):02d}-{1 + (i%28):02d}", 100 + i*0.2, 102 + i*0.2, 98 + i*0.2, 101 + i*0.2, 5000)
            for i in range(120)
        ]
        database.insert_prices_batch(prices)

        with patch("backtest.getData"):
            res = backtest.run_backtest("SIM", initial_capital=10000.0)
            self.assertIn("strategy_return_pct", res)
            self.assertIn("benchmark_return_pct", res)
            self.assertIn("max_drawdown_pct", res)
            self.assertIn("total_trades", res)


if __name__ == "__main__":
    unittest.main()
