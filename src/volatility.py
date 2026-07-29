import math

import numpy as np
import pandas as pd
import yfinance as yf

from database import create_connection, insert_volatility


# gets the last 21 days of close prices from db and computes the 20-day annualized
# historical volatility
def calc_20day_hv(ticker: str) -> float | None:
    connection = None
    try:
        connection = create_connection()
        query = """
            SELECT close FROM stock_prices
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT 21
        """

        df = pd.read_sql_query(query, connection, params=(ticker,))

        if len(df) < 21:
            return None

        close = df['close'].values[::-1]

        log_returns = np.log(np.array(close[1:]) / np.array(close[:-1]))

        daily_std = np.std(log_returns, ddof=1)

        annualized_hv = daily_std * math.sqrt(252)

        return float(annualized_hv)

    finally:
        if connection:
            connection.close()

# computes 52-week high/low 20-day HV range to proxy IV Rank
def calc_iv_rank_proxy(ticker: str, current_iv: float) -> float | None:
    connection = None
    try:
        connection = create_connection()
        # Fetch 272 trading days (252 days + 20-day window for initial HV)
        query = """
            SELECT date, close FROM stock_prices
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT 272
        """

        df = pd.read_sql_query(query, connection, params=(ticker,))

        if len(df) < 272:
            return None

        # Sort chronologically for rolling window calculation
        df = df.sort_values("date").reset_index(drop=True)

        # Log returns & rolling 20-day HV
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["hv_20"] = df["log_return"].rolling(window=20).std(ddof=1) * math.sqrt(252)

        hv_series = df["hv_20"].dropna()

        min_hv = hv_series.min()
        max_hv = hv_series.max()

        if max_hv == min_hv:
            return 50.0

        # IV Rank formula
        iv_rank = ((current_iv - min_hv) / (max_hv - min_hv)) * 100.0

        # Keep between 0.0 and 100.0
        return float(np.clip(iv_rank, 0.0, 100.0))

    except Exception as e:
        print(f"Error calculating IV Rank proxy for {ticker}: {e}")
        return None
    finally:
        if connection:
            connection.close()

# gets the live yfinance options for the closest expiration and gets the iv for the atm calls
def atm_iv(ticker: str) -> float | None:
    try:
        stock = yf.Ticker(ticker)

        current_price = stock.fast_info['lastPrice']

        expirations = stock.options
        if not expirations:
            print("no options found")
            return None

        target_expiration = expirations[0]
        options_chain = stock.option_chain(target_expiration)
        calls = options_chain.calls

        if calls.empty:
            return None

        if isinstance(calls.columns, pd.MultiIndex):
            calls.columns = calls.columns.get_level_values(0)

        calls['strike_diff'] = (calls['strike'] - current_price).abs()
        atm_option = calls.loc[calls['strike_diff'].idxmin()]

        atm_iv = atm_option['impliedVolatility']
        return float(atm_iv)

    except Exception:
        print("Error fetching options")
        return None

# this is for organizing the calculations and compares all of it in the database
def vol_scan(ticker : str):
    hv_20 = calc_20day_hv(ticker)
    iv_atm = atm_iv(ticker)

    if hv_20 is None or iv_atm is None:
        print(f"Skipping {ticker}: doesnt have metrics")
        return

    iv_rank = 0.0

    insert_volatility(ticker, hv_20, iv_atm, iv_rank)
    print(f"[{ticker}] Snapshot saved -> 20D HV: {hv_20:.2%}, ATM IV: {iv_atm:.2%}")
