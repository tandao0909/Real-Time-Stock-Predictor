import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.base import BaseEstimator
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from sklearn.neural_network import MLPRegressor


def prepare_ARIMA_data(
    X: pd.DataFrame, y: pd.Series, predict_ticker: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_valid, y_train, y_valid = prepare_train_valid_data(X, y)

    ARIMA_columns = [
        feature
        for feature in X_train.columns.to_list()
        if predict_ticker not in feature
    ]
    X_train_ARIMA = X_train.loc[:, ARIMA_columns]
    X_valid_ARIMA = X_valid.loc[:, ARIMA_columns]
    return X_train_ARIMA, X_valid_ARIMA, y_train, y_valid


def prepare_LSTM_data(
    X: pd.DataFrame, y: pd.Series, seq_len: int = 2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_valid, y_train, y_valid = prepare_train_valid_data(X, y)

    y_train_LSTM, y_valid_LSTM = np.array(y_train)[seq_len - 1 :], np.array(y_valid)
    X_train_LSTM = np.zeros([X_train.shape[0] + 1 - seq_len, seq_len, X_train.shape[1]])
    X_valid_LSTM = np.zeros([X_valid.shape[0], seq_len, X_valid.shape[1]])

    for i in range(seq_len):
        X_train_LSTM[:, i, :] = np.array(X_train)[
            i : X_train.shape[0] + i + 1 - seq_len, :
        ]
        X_valid_LSTM[:, i, :] = np.array(X)[
            X_train.shape[0] + i - 1 : X.shape[0] + i + 1 - seq_len, :
        ]

    return X_train_LSTM, X_valid_LSTM, y_train_LSTM, y_valid_LSTM


def prepare_train_valid_data(
    X: pd.DataFrame, y: pd.Series, valid_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_size = int(len(X) * (1 - valid_size))

    X_train, X_valid = X[:train_size], X[train_size:]
    y_train, y_valid = y[:train_size], y[train_size:]

    return X_train, X_valid, y_train, y_valid


def prepare_models(
    boosting: bool = True, bagging: bool = True
) -> list[tuple[str, BaseEstimator]]:
    models = []
    models.append(("Linear", LinearRegression()))
    models.append(("Lasso", Lasso()))
    models.append(("Elastic", ElasticNet()))

    models.append(("KNN", KNeighborsRegressor()))
    models.append(("CART", DecisionTreeRegressor()))
    models.append(("SVR", SVR()))

    models.append(("MLP", MLPRegressor()))
    if boosting:
        models.append(("ABR", AdaBoostRegressor()))
        models.append(("GBR", GradientBoostingRegressor()))
    if bagging:
        models.append(("RFR", RandomForestRegressor()))
        models.append(("ETR", ExtraTreesRegressor()))
    return models
