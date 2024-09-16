import yfinance as yf
import pandas as pd
import re

AMOUNT_OF_DATE = "10y"

def format_index(index: str) -> str:
    """
    Reformat index return by yfinance.

    :param index: The index, which has the form of "yyyy-mm-dd 00:00:00-04:00"
    :return: The index, strip away the string " 00:00:00-04:00"
    """
    return re.sub(r" .+", "", index)

def historical_stocks_data(stocks: list[str]) -> pd.DataFrame:
    """
    Return a pandas DataFrame contains the close price of all the stocks, back from 10 years ago.

    :param stocks: List of stock tickers
    :return: a pandas DataFrame contains the result
    """
    series = []
    for name in stocks:
        stock:yf.Ticker = yf.Ticker(name)
        close_data:pd.DataFrame = stock.history(AMOUNT_OF_DATE)["Close"]
        close_data.index = close_data.index.astype(str).to_series().apply(format_index)
        close_data.rename(name, inplace=True)
        series.append(close_data)
    combined_df = pd.concat(series, axis=1)
    combined_df.index = pd.to_datetime(combined_df.index, format="%Y-%m-%d")
    return combined_df

if __name__ == "__main__":
    test_close_data = historical_stocks_data(["AAPL", "MSFT", "IBM"])
    print(test_close_data)
