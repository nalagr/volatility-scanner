import pandas as pd
import yfinance as yf

from database import insert_prices_batch, insert_stock


# This function gets the stocks data given a ticker symbol, start date, and end endDate
# Ex: getData("NVDA", "2020-01-01", "2021-01-01") -> returns dataframe that gets stored in database
def getData(ticker:str, startDate:str, endDate:str) -> pd.DataFrame:

    # get data from yfinance
    stock_data = yf.download(ticker, start=startDate, end=endDate)

    # case where yfinance returns no data for the given ticker
    if stock_data is None or stock_data.empty:
        print(f"No data was found for {ticker} ")
        return pd.DataFrame()

    # insert the stock
    insert_stock(ticker)

    batch_data = []

    for date, row in stock_data.iterrows():
        # formats the date
        date_str = str(date)[:10]

        # getting the numeric values of the open, high, low, close, vol
        try:
            record = (
                ticker,
                date_str,
                float(row['Open'].item() if hasattr(row['Open'], 'item') else row['Open']),
                float(row['High'].item() if hasattr(row['High'], 'item') else row['High']),
                float(row['Low'].item() if hasattr(row['Low'], 'item') else row['Low']),
                float(row['Close'].item() if hasattr(row['Close'], 'item') else row['Close']),
                int(row['Volume'].item() if hasattr(row['Volume'], 'item') else row['Volume'])
            )
            batch_data.append(record)
        except Exception:
            # skip empty
            continue

    # execute the batch insertion
    if batch_data:
        insert_prices_batch(batch_data)
        print(f"Successfully processed and stored {len(batch_data)} rows for {ticker}")

    return stock_data