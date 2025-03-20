import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates

from constants import STOCK_DATA_FOLDER

def plot_test_fine_tuned_ARIMA(
    X: pd.DataFrame, y: pd.Series, predict_ticker: str
) -> None:
    ARIMA_fine_tuned = fine_tune_ARIMA(X, y, predict_ticker)

    X_train_ARIMA, X_valid_ARIMA, y_train, y_valid = prepare_ARIMA_data(
        X, y, predict_ticker
    )

    predicted_tuned = ARIMA_fine_tuned.predict(
        start=len(X_train_ARIMA) - 1, end=len(X) - 1, exog=X_valid_ARIMA
    )[1:]
    predicted_tuned.index = y_valid.index

    plt.figure()

    plt.plot(np.exp(y_valid).cumprod(), label="actual")
    plt.plot(np.exp(predicted_tuned).cumprod(), label="predict")
    plt.gca().xaxis.set_major_locator(
        dates.MonthLocator(interval=1)
    )  # Change interval as needed
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%Y-%m"))
    plt.legend()

    plt.savefig(f"{STOCK_DATA_FOLDER}/{predict_ticker}/{predict_ticker}_backtest.png")
    # plt.show()


def plot_K_folds_comparison(
    names: list[str], k_fold_results: list[np.ndarray], predict_ticker: str
) -> None:
    plt.figure(figsize=(15, 8))

    plt.boxplot(k_fold_results)

    plt.title("KFold result")
    plt.xticks(ticks=range(len(names) + 1), labels=[0] + names)

    plt.savefig(f"{STOCK_DATA_FOLDER}/{predict_ticker}/{predict_ticker}_k_folds.png")
    # plt.show()


def plot_train_valid_comparison(
    names: list[str],
    train_results: list[float],
    valid_results: list[float],
    predict_ticker: str,
) -> None:
    plt.figure(figsize=(15, 8))

    index = np.arange(len(names))
    width = 0.3

    plt.bar(index - width / 2, train_results, width=width, label="Train error")
    plt.bar(index + width / 2, valid_results, width=width, label="valid error")
    plt.xticks(ticks=range(len(train_results)), labels=names)

    plt.title("Algorithm Comparison")
    plt.legend()

    plt.savefig(f"{STOCK_DATA_FOLDER}/{predict_ticker}/{predict_ticker}_algo_comparison.png")
    # plt.show()
