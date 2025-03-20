"""
Contains Data Processing functions.

Only need to import `process_data`.
"""
import numpy as np
import pandas as pd

from constants import RETURN_PERIOD, SELF_BUSINESS_RETURNS_DAYS


def process_stock_data(
    stock_tickers: list[str], stock_data: pd.DataFrame, predict_ticker: str
) -> pd.DataFrame:
    stock_tickers_copy = stock_tickers.copy()
    stock_tickers_copy.remove(predict_ticker)
    return np.log(stock_data.loc[:, stock_tickers]).diff(RETURN_PERIOD)


def process_index_data(index_data: pd.DataFrame) -> pd.DataFrame:
    return np.log(index_data).diff(RETURN_PERIOD)


def process_currency_data(currency_data: pd.DataFrame) -> pd.DataFrame:
    return np.log(currency_data).diff(RETURN_PERIOD)


def process_dependent_data(stock_data: pd.DataFrame, predict_ticker: str) -> pd.Series:
    y = (
        np.log(stock_data.loc[:, predict_ticker])
        .diff(RETURN_PERIOD)
        .shift(-RETURN_PERIOD)
    )
    y.name += "_pred"
    return y


def process_endogenous_data(
    stock_data: pd.DataFrame, predict_ticker: str
) -> pd.DataFrame:
    X4 = pd.concat(
        [
            np.log(stock_data.loc[:, predict_ticker]).diff(i)
            for i in SELF_BUSINESS_RETURNS_DAYS
        ],
        axis=1,
    ).dropna()
    X4.columns = [f"{predict_ticker}_{i}DR" for i in SELF_BUSINESS_RETURNS_DAYS]
    return X4


def process_data(
    stock_data: pd.DataFrame,
    index_data: pd.DataFrame,
    currency_data: pd.DataFrame,
    stock_tickers: list[str],
    predict_ticker: str,
) -> tuple[pd.DataFrame, pd.Series]:
    X1 = process_stock_data(stock_tickers, stock_data, predict_ticker)
    X2 = process_index_data(index_data)
    X3 = process_currency_data(currency_data)
    X4 = process_endogenous_data(stock_data, predict_ticker)
    y = process_dependent_data(stock_data, predict_ticker)

    X = pd.concat([X1, X2, X3, X4], axis=1).dropna()

    dataset = pd.concat([X, y], axis=1).dropna().iloc[::RETURN_PERIOD, :]

    X = dataset.loc[:, X.columns]
    y = dataset.loc[:, y.name]

    return X, y
