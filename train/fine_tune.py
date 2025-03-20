import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

from prepare import prepare_ARIMA_data

def fine_tune_ARIMA(X: pd.DataFrame, y: pd.Series, predict_ticker: str) -> ARIMA:
    X_train_ARIMA, _, y_train, _ = prepare_ARIMA_data(X, y, predict_ticker)
    ARIMA_fine_tuned = ARIMA(
        endog=y_train,
        exog=X_train_ARIMA,
        order=find_best_order_ARIMA(X, y, predict_ticker),
    ).fit()

    return ARIMA_fine_tuned


def find_best_order_ARIMA(
    X: pd.DataFrame,
    y: pd.Series,
    predict_ticker: str,
    p_values: tuple[int, int, int] = (0, 1, 2),
    d_values: tuple[int, int, int] = (0, 1, 2),
    q_values: tuple[int, int, int] = (0, 1, 2),
) -> tuple[int, int, int]:
    X_train_ARIMA, _, y_train, _ = prepare_ARIMA_data(X, y, predict_ticker)

    def evaluate_ARIMA_model(order: tuple[int, int, int]) -> float:
        ARIMA_model = ARIMA(endog=y_train, exog=X_train_ARIMA, order=order).fit()
        error = mean_squared_error(y_train, ARIMA_model.fittedvalues)
        return error

    best_score, best_order = float("inf"), (0, 0, 0)

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                mse = evaluate_ARIMA_model(order)
                if mse < best_score:
                    best_score, best_order = mse, order
    return best_order
