import sqlite3
from pathlib import Path

# path for database, run from anywhere
DB_PATH = Path(__file__).parent.parent / "data" / "trading.db"


# creating the connection with WAL mode and busy timeout
def create_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10.0)
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA busy_timeout=5000;")
    connection.execute("PRAGMA foreign_keys = ON;")
    connection.row_factory = sqlite3.Row
    return connection


# creating the tables
def create_tables():
    connection = None
    try:
        connection = create_connection()
        cursor = connection.cursor()

        # table for storing the stocks and the date last fetched
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stocks (
                ticker TEXT PRIMARY KEY,
                sector TEXT,
                add_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_fetched TIMESTAMP
            )"""
        )

        # table for storing the stocks data. links to the stocks table via foreign key reference
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                VOLUME INTEGER,
                FOREIGN KEY (ticker) REFERENCES stocks(ticker),
                UNIQUE(ticker,date)
            )"""
        )

        # table for storing the volatility snapshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS volatility_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hv_20day REAL,
                atm_iv REAL,
                iv_rank REAL,
                FOREIGN KEY (ticker) REFERENCES stocks(ticker)
            )"""
        )

        # table for active position tracking across strategy lifecycle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PHASE1_OPEN', -- 'PHASE1_OPEN', 'PHASE2_HEDGED', 'CLOSED'
                entry_date DATE NOT NULL,
                long_strike REAL NOT NULL,
                long_expiry DATE NOT NULL,
                long_cost REAL NOT NULL,
                short_strike REAL,
                short_expiry DATE,
                short_credit REAL,
                net_cost_basis REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticker) REFERENCES stocks(ticker)
            )"""
        )

        # indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON stock_prices(ticker, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vol_ticker_timestamp ON volatility_snapshots(ticker, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker)")

        connection.commit()
    finally:
        if connection:
            connection.close()


def insert_stock(ticker: str, sector: str | None = None):
    connection = None
    try:
        connection = create_connection()
        connection.execute(
            "INSERT OR IGNORE INTO stocks (ticker, sector) VALUES (?, ?)",
            (ticker, sector)
        )
        connection.commit()
    finally:
        if connection:
            connection.close()


def insert_price(ticker: str, date: str, open_price: float, high: float, low: float, close: float, volume: int):
    connection = None
    try:
        connection = create_connection()
        connection.execute("INSERT OR IGNORE INTO stocks (ticker) VALUES (?)", (ticker,))
        connection.execute("""
            INSERT INTO stock_prices (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ticker, date, open_price, high, low, close, volume))
        connection.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        if connection:
            connection.close()


def insert_prices_batch(prices_list: list[tuple]):
    connection = None
    try:
        connection = create_connection()
        tickers = list({p[0] for p in prices_list})
        connection.executemany("INSERT OR IGNORE INTO stocks (ticker) VALUES (?)", [(t,) for t in tickers])
        connection.executemany("""
            INSERT OR IGNORE INTO stock_prices (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, prices_list)
        connection.commit()
    finally:
        if connection:
            connection.close()


def insert_volatility(ticker: str, hv_20day: float, atm_iv: float, iv_rank: float):
    connection = None
    try:
        connection = create_connection()
        connection.execute("INSERT OR IGNORE INTO stocks (ticker) VALUES (?)", (ticker,))
        connection.execute("""
            INSERT INTO volatility_snapshots (ticker, hv_20day, atm_iv, iv_rank)
            VALUES (?, ?, ?, ?)
        """, (ticker, hv_20day, atm_iv, iv_rank))
        connection.commit()
    finally:
        if connection:
            connection.close()


def get_latest_price_date(ticker: str) -> str | None:
    connection = None
    try:
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT MAX(date) FROM stock_prices WHERE ticker = ?", (ticker,))
        row = cursor.fetchone()
        if row and row[0]:
            return str(row[0])
        return None
    finally:
        if connection:
            connection.close()


def get_historical_iv_snapshots(ticker: str) -> list[float]:
    connection = None
    try:
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT atm_iv FROM volatility_snapshots WHERE ticker = ? AND atm_iv IS NOT NULL AND atm_iv > 0",
            (ticker,)
        )
        return [float(r[0]) for r in cursor.fetchall()]
    finally:
        if connection:
            connection.close()


def insert_position(
    ticker: str,
    entry_date: str,
    long_strike: float,
    long_expiry: str,
    long_cost: float,
    notes: str | None = None
) -> int:
    connection = None
    try:
        connection = create_connection()
        cursor = connection.cursor()
        # Guarantee parent stock record exists to satisfy foreign key constraint
        cursor.execute("INSERT OR IGNORE INTO stocks (ticker) VALUES (?)", (ticker,))
        net_cost_basis = long_cost
        cursor.execute("""
            INSERT INTO positions (
                ticker, status, entry_date, long_strike, long_expiry,
                long_cost, net_cost_basis, notes
            )
            VALUES (?, 'PHASE1_OPEN', ?, ?, ?, ?, ?, ?)
        """, (ticker, entry_date, long_strike, long_expiry, long_cost, net_cost_basis, notes))
        connection.commit()
        return cursor.lastrowid
    finally:
        if connection:
            connection.close()


def update_position_hedge(
    position_id: int,
    short_strike: float,
    short_expiry: str,
    short_credit: float,
    notes: str | None = None
) -> bool:
    """Hedges an open position into PHASE2_HEDGED and reduces net cost basis."""
    connection = None
    try:
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT long_cost, net_cost_basis, notes FROM positions WHERE id = ?", (position_id,))
        row = cursor.fetchone()
        if not row:
            return False

        long_cost = float(row["long_cost"])
        # Support cumulative credits if hedged or rolled multiple times
        current_basis = float(row["net_cost_basis"]) if row["net_cost_basis"] is not None else long_cost
        net_cost_basis = round(current_basis - short_credit, 2)

        combined_notes = notes
        if row["notes"] and notes:
            combined_notes = f"{row['notes']} | {notes}"
        elif row["notes"] and not notes:
            combined_notes = row["notes"]

        cursor.execute("""
            UPDATE positions
            SET status = 'PHASE2_HEDGED',
                short_strike = ?,
                short_expiry = ?,
                short_credit = ?,
                net_cost_basis = ?,
                notes = ?
            WHERE id = ?
        """, (short_strike, short_expiry, short_credit, net_cost_basis, combined_notes, position_id))
        connection.commit()
        return cursor.rowcount > 0
    finally:
        if connection:
            connection.close()


def roll_position_hedge(
    position_id: int,
    new_short_strike: float,
    new_short_expiry: str,
    new_short_credit: float,
    notes: str | None = None
) -> bool:
    """Rolls a Phase 2 short call hedge forward to a new strike/expiry and accumulates credit."""
    roll_note = f"Rolled hedge to ${new_short_strike:.2f} exp {new_short_expiry} (+${new_short_credit:.2f})"
    if notes:
        roll_note += f": {notes}"
    return update_position_hedge(position_id, new_short_strike, new_short_expiry, new_short_credit, roll_note)


def unhedge_position(position_id: int, notes: str | None = None) -> bool:
    """Reverts a hedged position back to PHASE1_OPEN when short call expires or is closed."""
    connection = None
    try:
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT notes FROM positions WHERE id = ?", (position_id,))
        row = cursor.fetchone()
        if not row:
            return False

        unhedge_note = notes or "Short call expired worthless / unhedged"
        combined_notes = f"{row['notes']} | {unhedge_note}" if row["notes"] else unhedge_note

        cursor.execute("""
            UPDATE positions
            SET status = 'PHASE1_OPEN',
                short_strike = NULL,
                short_expiry = NULL,
                short_credit = NULL,
                notes = ?
            WHERE id = ?
        """, (combined_notes, position_id))
        connection.commit()
        return cursor.rowcount > 0
    finally:
        if connection:
            connection.close()


def close_position(position_id: int, notes: str | None = None) -> bool:
    connection = None
    try:
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT notes FROM positions WHERE id = ?", (position_id,))
        row = cursor.fetchone()
        if not row:
            return False

        combined_notes = notes
        if row["notes"] and notes:
            combined_notes = f"{row['notes']} | {notes}"
        elif row["notes"] and not notes:
            combined_notes = row["notes"]

        cursor.execute("""
            UPDATE positions
            SET status = 'CLOSED',
                notes = ?
            WHERE id = ?
        """, (combined_notes, position_id))
        connection.commit()
        return cursor.rowcount > 0
    finally:
        if connection:
            connection.close()


def get_positions(status: str | None = None) -> list[dict]:
    connection = None
    try:
        connection = create_connection()
        cursor = connection.cursor()
        if status:
            cursor.execute("SELECT * FROM positions WHERE status = ? ORDER BY id DESC", (status,))
        else:
            cursor.execute("SELECT * FROM positions ORDER BY id DESC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        if connection:
            connection.close()


def get_open_positions() -> list[dict]:
    """Returns active Phase 1 unhedged positions."""
    return get_positions(status="PHASE1_OPEN")


def get_hedged_positions() -> list[dict]:
    """Returns active Phase 2 hedged positions."""
    return get_positions(status="PHASE2_HEDGED")


def update_snapshot_iv_rank(snapshot_id: int, iv_rank: float) -> bool:
    """Updates IV rank for an existing volatility snapshot."""
    connection = None
    try:
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute(
            "UPDATE volatility_snapshots SET iv_rank = ? WHERE id = ?",
            (round(iv_rank, 2), snapshot_id)
        )
        connection.commit()
        return cursor.rowcount > 0
    finally:
        if connection:
            connection.close()


def get_zero_rank_snapshots() -> list[dict]:
    """Returns snapshots with missing or zero IV rank."""
    connection = None
    try:
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute("""
            SELECT id, ticker, timestamp, hv_20day, atm_iv, iv_rank
            FROM volatility_snapshots
            WHERE (iv_rank = 0.0 OR iv_rank IS NULL) AND atm_iv IS NOT NULL AND atm_iv > 0.01
            ORDER BY id ASC
        """)
        return [dict(r) for r in cursor.fetchall()]
    finally:
        if connection:
            connection.close()
