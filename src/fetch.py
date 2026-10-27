from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

from database import insert_prices_batch, insert_stock, get_latest_price_date


def getData(
    ticker: str,
    startDate: str | None = None,
    endDate: str | None = None,
    force_full: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetches daily stock prices for a ticker with smart incremental sync.
    If the ticker already has data in SQLite and is up-to-date, fetching is skipped.
    If it is a few days behind, only the missing date range is requested.
    If force_full=True or no data exists, fetches a full 400-day window.
    """
    today_dt = datetime.now().date()
    today_str = today_dt.strftime("%Y-%m-%d")

    target_end_dt = datetime.strptime(endDate, "%Y-%m-%d").date() if endDate else today_dt
    default_start_dt = (
        datetime.strptime(startDate, "%Y-%m-%d").date()
        if startDate
        else (today_dt - timedelta(days=400))
    )

    fetch_start_dt = default_start_dt
    is_incremental = False

    if not force_full and startDate is None:
        latest_date_str = get_latest_price_date(ticker)
        if latest_date_str:
            try:
                latest_dt = datetime.strptime(latest_date_str[:10], "%Y-%m-%d").date()

                # 1. If latest in DB is already >= target_end_dt, nothing to fetch
                if latest_dt >= target_end_dt:
                    if verbose:
                        print(f"[{ticker}] Up-to-date (latest in DB: {latest_date_str}). Skipping download.")
                    return pd.DataFrame()

                # 2. Weekend check: if Saturday/Sunday and Friday's close is already stored
                delta_days = (today_dt - latest_dt).days
                if today_dt.weekday() == 5 and delta_days <= 1:  # Saturday
                    if verbose:
                        print(f"[{ticker}] Weekend up-to-date (Friday: {latest_date_str}). Skipping download.")
                    return pd.DataFrame()
                elif today_dt.weekday() == 6 and delta_days <= 2:  # Sunday
                    if verbose:
                        print(f"[{ticker}] Weekend up-to-date (Friday: {latest_date_str}). Skipping download.")
                    return pd.DataFrame()

                # 3. Incremental: resume from day after latest stored record
                next_day_dt = latest_dt + timedelta(days=1)
                if next_day_dt > target_end_dt:
                    if verbose:
                        print(f"[{ticker}] Up-to-date (latest in DB: {latest_date_str}). Skipping download.")
                    return pd.DataFrame()

                fetch_start_dt = next_day_dt
                is_incremental = True
                if verbose:
                    print(f"[{ticker}] Incremental sync: {fetch_start_dt} -> {target_end_dt} (latest: {latest_date_str})")
            except Exception as e:
                if verbose:
                    print(f"[{ticker}] Could not parse latest date '{latest_date_str}': {e}. Performing full sync.")
                fetch_start_dt = default_start_dt
        else:
            if verbose:
                print(f"[{ticker}] Initial fetch: {fetch_start_dt} -> {target_end_dt}")
    else:
        if verbose:
            print(f"[{ticker}] Full fetch: {fetch_start_dt} -> {target_end_dt}")

    fetch_start_str = fetch_start_dt.strftime("%Y-%m-%d")
    # yfinance 'end' parameter is exclusive; add 1 day so target_end_dt is included
    yf_end_str = (target_end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        stock_data = yf.download(
            ticker,
            start=fetch_start_str,
            end=yf_end_str,
            progress=False
        )
    except Exception as e:
        if verbose:
            print(f"[{ticker}] Error downloading from yfinance: {e}")
        return pd.DataFrame()

    if stock_data is None or stock_data.empty:
        if verbose and not is_incremental:
            print(f"[{ticker}] No price data returned for {fetch_start_str} to {target_end_dt}")
        return pd.DataFrame()

    # Flatten multi-index columns if returned by newer yfinance
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)

    insert_stock(ticker)

    batch_data = []
    for date, row in stock_data.iterrows():
        date_str = str(date)[:10]
        try:
            o = float(row['Open'].item() if hasattr(row['Open'], 'item') else row['Open'])
            h = float(row['High'].item() if hasattr(row['High'], 'item') else row['High'])
            l = float(row['Low'].item() if hasattr(row['Low'], 'item') else row['Low'])
            c = float(row['Close'].item() if hasattr(row['Close'], 'item') else row['Close'])
            v = int(row['Volume'].item() if hasattr(row['Volume'], 'item') else row['Volume'])

            if pd.isna(o) or pd.isna(c):
                continue

            batch_data.append((ticker, date_str, o, h, l, c, v))
        except Exception:
            continue

    if batch_data:
        insert_prices_batch(batch_data)
        if verbose:
            print(f"[{ticker}] Stored {len(batch_data)} new price records.")

    return stock_data