import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas_datareader.data as web
import pandas as pd

from constants import YEARS_TO_FORECAST, STOCK_DATA_FOLDER
from yahoo import historical_stocks_data


def extract_data(
    stock_tickers: list[str],
    currency_tickers: list[str],
    index_tickers: list[str],
    predict_ticker: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    begin_date = datetime.now() - relativedelta(years=YEARS_TO_FORECAST, days=-1)
    if os.path.isdir(f"{STOCK_DATA_FOLDER}/{predict_ticker}/data"):
        currency_data = pd.read_csv(
            f"{STOCK_DATA_FOLDER}/{predict_ticker}/data/{predict_ticker}_currency_data.csv",
            index_col="DATE",
        )
        index_data = pd.read_csv(
            f"{STOCK_DATA_FOLDER}/{predict_ticker}/data/{predict_ticker}_index_data.csv", index_col="DATE"
        )
        stock_data = pd.read_csv(
            f"{STOCK_DATA_FOLDER}/{predict_ticker}/data/{predict_ticker}_stock_data.csv", index_col="Date"
        )
    else:
        print(f"Not found data of {predict_ticker} locally, start downloading from FRED...")
        os.makedirs(f"{STOCK_DATA_FOLDER}/{predict_ticker}/data")
        currency_data: pd.DataFrame = web.DataReader(
            currency_tickers, "fred", start=begin_date, end=datetime.now()
        )
        index_data: pd.DataFrame = web.DataReader(
            index_tickers, "fred", start=begin_date, end=datetime.now()
        )
        stock_data: pd.DataFrame = historical_stocks_data(stock_tickers)
        currency_data.to_csv(
            f"{STOCK_DATA_FOLDER}/{predict_ticker}/data/{predict_ticker}_currency_data.csv"
        )
        index_data.to_csv(f"{STOCK_DATA_FOLDER}/{predict_ticker}/data/{predict_ticker}_index_data.csv")
        stock_data.to_csv(f"{STOCK_DATA_FOLDER}/{predict_ticker}/data/{predict_ticker}_stock_data.csv")
    return stock_data, currency_data, index_data
