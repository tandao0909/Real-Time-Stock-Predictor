# pylint: disable=redefined-outer-name
# pylint: disable=invalid-name

import warnings
import os
from pickle import dump

from statsmodels.tsa.arima.model import ARIMA
from sklearn.base import BaseEstimator

from extract import extract_data
from prepare import prepare_models
from process import process_data
from train_models import train_model_list
from plotting import (
    plot_K_folds_comparison,
    plot_test_fine_tuned_ARIMA,
    plot_train_valid_comparison,
)
from fine_tune import fine_tune_ARIMA

warnings.filterwarnings("ignore")


def save_model(model: ARIMA | BaseEstimator, predict_ticker: str) -> None:
    if not os.path.exists(f"{predict_ticker}"):
        os.makedirs(f"{predict_ticker}")
    filename = f"{predict_ticker}/{predict_ticker}.pkl"
    with open(filename, "wb") as f:
        dump(model, f)


if __name__ == "__main__":
    stock_tickers = ["MSFT", "GOOGL", "IBM", "AMZN", "AAPL", "NVDA"]
    currency_tickers = ["DEXJPUS", "DEXUSUK"]
    index_tickers = ["SP500", "DJIA", "NASDAQ100"]

    for predict_ticker in stock_tickers:
        print(f"Processing {predict_ticker}")
        stock_data, currency_data, index_data = extract_data(
            stock_tickers, currency_tickers, index_tickers, predict_ticker
        )

        X, y = process_data(
            stock_data, index_data, currency_data, stock_tickers, predict_ticker
        )

        names, k_fold_results, train_results, valid_results = train_model_list(
            models=prepare_models(), X=X, y=y, predict_ticker=predict_ticker
        )

        plot_K_folds_comparison(names, k_fold_results, predict_ticker)

        plot_train_valid_comparison(names, train_results, valid_results, predict_ticker)

        plot_test_fine_tuned_ARIMA(X, y, predict_ticker)

        save_model(fine_tune_ARIMA(X, y, predict_ticker), predict_ticker)
