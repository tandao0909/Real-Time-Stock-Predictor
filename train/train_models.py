import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer

from statsmodels.tsa.arima.model import ARIMA

from constants import NUM_FOLDS, SCORING, SEED
from prepare import prepare_train_valid_data, prepare_LSTM_data, prepare_ARIMA_data

def train_model_list(
    models: list[tuple[str, BaseEstimator]],
    X: pd.DataFrame,
    y: pd.Series,
    train_ARIMA: bool = True,
    train_LSTM: bool = True,
    predict_ticker: str = "",
) -> tuple[list[str], list[np.ndarray], list[float], list[float]]:
    """
    This function will always train a list of ML models: Linear, Tree-based, MLP, AdaBoost, GradientBoost, Random Forest, and Extra Tree.

    You can use the train_ARIMA and train_LSTM to control whether to use ARIMA and LSTM. They are true by default.
    """
    names = []
    k_fold_results = []
    valid_results = []
    train_results = []

    model_stats_df = pd.DataFrame(
        columns=["Name", "Mean", "Std", "Train MSE", "Valid MSE"]
    )

    X_train, X_valid, y_train, y_valid = prepare_train_valid_data(X, y)
    for name, model in models:
        names.append(name)

        k_fold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        cv_result = -1 * cross_val_score(
            model, X_train, y_train, scoring=SCORING, cv=k_fold
        )
        k_fold_results.append(cv_result)

        trained_model = model.fit(X_train, y_train)
        train_result = mean_squared_error(trained_model.predict(X_train), y_train)
        train_results.append(train_result)

        valid_result = mean_squared_error(trained_model.predict(X_valid), y_valid)
        valid_results.append(valid_result)

        new_df = pd.DataFrame(
            {
                "Name": [name],
                "Mean": [cv_result.mean()],
                "Std": [cv_result.std()],
                "Train MSE": [train_result],
                "Valid MSE": [valid_result],
            }
        )
        model_stats_df = pd.concat([model_stats_df, new_df], ignore_index=True)
    model_stats_df.set_index("Name", inplace=True)
    print(model_stats_df)

    if train_ARIMA:
        _, error_ARIMA, error_valid_ARIMA = train_ARIMA_model(X, y, predict_ticker)
        names.append("ARIMA")
        train_results.append(error_ARIMA)
        valid_results.append(error_valid_ARIMA)

    if train_LSTM:
        _, error_LSTM, error_valid_LSTM = train_LSTM_model(X, y)
        names.append("LSTM")
        train_results.append(error_LSTM)
        valid_results.append(error_valid_LSTM)

    return names, k_fold_results, train_results, valid_results


def train_ARIMA_model(
    X: pd.DataFrame, y: pd.Series, predict_ticker: str
) -> tuple[ARIMA, float, float]:
    X_train_ARIMA, X_valid_ARIMA, y_train, y_valid = prepare_ARIMA_data(
        X, y, predict_ticker
    )

    ARIMA_model = ARIMA(endog=y_train, exog=X_train_ARIMA, order=(1, 0, 0))
    ARIMA_model = ARIMA_model.fit()

    error_ARIMA = mean_squared_error(y_train, ARIMA_model.fittedvalues)
    predict = ARIMA_model.predict(
        start=len(X_train_ARIMA) - 1, end=len(X) - 1, exog=X_valid_ARIMA
    )[1:]
    error_valid_ARIMA = mean_squared_error(y_valid, predict)

    return ARIMA_model, error_ARIMA, error_valid_ARIMA


def train_LSTM_model(
    X: pd.DataFrame, y: pd.Series, plot_history: bool = True
) -> tuple[Sequential, float, float]:
    X_train_LSTM, X_valid_LSTM, y_train_LSTM, y_valid_LSTM = prepare_LSTM_data(X, y)

    def create_LSTMmodel(neurons=50, learning_rate=0.01, momentum=0) -> Sequential:
        model = Sequential(
            [
                InputLayer([X_train_LSTM.shape[1], X_train_LSTM.shape[2]]),
                LSTM(neurons),
                Dense(1),
            ]
        )
        model.compile(loss="mse", optimizer="adam")
        return model

    LSTMmodel = create_LSTMmodel(50, learning_rate=0.01, momentum=0)

    history = LSTMmodel.fit(
        X_train_LSTM,
        y_train_LSTM,
        validation_data=(X_valid_LSTM, y_valid_LSTM),
        epochs=330,
        batch_size=72,
        verbose=0,
        shuffle=False,
    )

    error_LSTM = mean_squared_error(y_train_LSTM, LSTMmodel.predict(X_train_LSTM))
    error_valid_LSTM = mean_squared_error(y_valid_LSTM, LSTMmodel.predict(X_valid_LSTM))

    if plot_history:
        plt.figure()

        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="val")

        plt.title("LSTM history")

        plt.legend()
        # plt.show()

    return LSTMmodel, error_LSTM, error_valid_LSTM
