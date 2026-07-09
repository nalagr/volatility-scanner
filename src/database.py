import sqlite3
from pathlib import Path

# path for database, run from anywhere
DB_PATH = Path(__file__).parent.parent / "data" / "trading.db"


# creating the connection
def create_connection():
    return sqlite3.connect(DB_PATH)


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

        # table for the storing the volatility snapshots
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

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker_date on stock_prices(ticker, date)")

        connection.commit()
    finally:
        if connection:
            connection.close()

# this functions takes the stock price data from a fetch and adds it to the stock table
# run from fetch to start tracking a stock
def insert_stock(ticker, sector=None):
    connection = None
    try:
        connection = create_connection()
        # inserts the stock data, if it already exists, does nothing
        connection.execute("INSERT OR IGNORE INTO stocks (ticker, sector) VALUES (?,?)",
                        (ticker, sector)
        )
        connection.commit()
    finally:
        if connection:
            connection.close()

# this function take the stock price data from a fetch and stores it in the stock_prices table
def insert_price(ticker, date, open_price, high, low, close, volume):
    connection = None
    try:
        connection = create_connection()
        connection.execute(""" INSERT INTO stock_prices
                           (ticker, date, open, high, low, close, VOLUME)
                           VALUES (?, ?, ?, ?, ?, ?, ?) """,
                           (ticker, date, open_price, high, low, close, volume))
        connection.commit()
    # this error catches any duplicate entries
    except sqlite3.IntegrityError:
        pass
    # closes even though an error might have occured
    finally:
        if connection:
            connection.close()

# this function inserts a large list of prices in a single transaction
def insert_prices_batch(prices_list):
    connection = None
    try:
        connection = create_connection()
        connection.executemany(""" INSERT OR IGNORE INTO stock_prices
                               (ticker, date, open, high, low, close, volume)
                               VALUES (?, ?, ?, ?, ?, ?, ?)
                               """, prices_list)
        connection.commit()
    finally:
        if connection:
            connection.close()

# this function will get the new volatility data points. run multiple times a day
def insert_volatility(ticker, hv_20day, atm_iv, iv_rank):
    connection = None
    try:
        connection =create_connection()
        connection.execute("""
                           INSERT INTO volatility_snapshots (ticker, hv_20day, atm_iv, iv_rank)
                           VALUES (?, ?, ?, ?)
                           """, (ticker, hv_20day, atm_iv, iv_rank))
        connection.commit()
    finally:
        if connection:
            connection.close()
