import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import requests

from database import create_tables
from fetch import getData
from volatility import vol_scan


def get_sp500_tickers() -> list[str]:
    # gets the current sp500 companies from wiki
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        html_data = io.StringIO(response.text)
        tables = pd.read_html(html_data)

        sp500_df = tables[0]
        tickers = sp500_df["Symbol"].str.replace(".", "-", regex=False).tolist()
        return tickers

    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AMD"]


def process_single_ticker(ticker: str, start_date: str, end_date: str) -> str:
    """Worker function to process a single ticker."""
    try:
        # 1. Fetch price history into SQLite
        getData(ticker, start_date, end_date)

        # 2. Calculate volatility metrics and store snapshot
        vol_scan(ticker)

        return f"[SUCCESS] Processed {ticker}"
    except Exception as e:
        return f"[SKIPPED] {ticker} due to error: {e}"


def main():
    print("Initializing Database...")
    create_tables()

    # gets the last 400 days of data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")

    print("Getting S&P 500 tickers...")
    tickers = get_sp500_tickers()
    total_tickers = len(tickers)
    print(
        f"Found {total_tickers} tickers. Date range: {start_date} to {end_date}\n"
    )

    # amount of workers so i dont get rate limited\
    max_workers = 7

    print(
        f"Starting parallel scan across {total_tickers} tickers using {max_workers} threads...\n"
    )

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all ticker tasks to the pool
        future_to_ticker = {
            executor.submit(
                process_single_ticker, ticker, start_date, end_date
            ): ticker
            for ticker in tickers
        }

        # Process results as they complete
        for future in as_completed(future_to_ticker):
            completed += 1
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                print(f"[{completed}/{total_tickers}] {result}")
            except Exception as e:
                print(f"[{completed}/{total_tickers}] [ERROR] {ticker}: {e}")

    print("\nScan completed successfully!")


if __name__ == "__main__":
    main()