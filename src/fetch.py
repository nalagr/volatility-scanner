import yfinance as yf
import pandas as pd

def getData(ticker:str, startDate:str, endDate:str) -> pd.DataFrame:
    # This function gets the stocks data given a ticker symbol, start date, and end endDate
    # Ex: getData("NVDA", "2020-01-01", "2021-01-01")

