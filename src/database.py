import sqlite3
from pathlib import Path

# path for database, run from anywhere
DB_PATH = Path(__file__).parent.parent / "data" / "trading.db"


# creating the connection
def create_connection():
    return sqlite3.connect(DB_PATH)


# creating the tables
def create_tables():
    connection = create_connection()
    cursor = connection.cursor()

    # table for storing the stocks and the date last fetched
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            ticker TEXT PRIMARY KEY,
            company_name TEXT,
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

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices+ticker+data on stock_prices(ticker, date)")

    connection.commit()
    connection.close()

# this functions adds a new stock to track
def insert_stock(ticker, company_name, sector=None):
    connection = create_connection()
    connection.execute("INSERT OR IGNORE INTO stocks (ticker, company_name, sector) VALUES (?,?,?)",
                       (ticker, company_name, sector)
    )
    connection.commit()
    connection.close()
