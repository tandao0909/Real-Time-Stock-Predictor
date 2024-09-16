# pylint: disable=redefined-outer-name
# pylint: disable=invalid-name

import warnings
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.base import BaseEstimator
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression, SelectKBest

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from pickle import dump

from yahoo import historical_stocks_data

warnings.filterwarnings("ignore")

YEARS_TO_FORECAST = 10
RETURN_PERIOD = 5
SELF_BUSINESS_RETURNS_DAYS = [5, 15, 30, 60]

NUM_FOLDS = 10
SEED = 42
SCORING = "neg_mean_squared_error"

def extract_data(
        stock_tickers:list[str],
        currency_tickers:list[str],
        index_tickers:list[str],
        predict_ticker:str
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    begin_date = datetime.now() - relativedelta(years=YEARS_TO_FORECAST, days=-1)
    if os.path.isdir(f"{predict_ticker}/data"):
        currency_data = pd.read_csv(f"./{predict_ticker}/data/currency_data.csv", index_col="DATE")
        index_data = pd.read_csv(f"./{predict_ticker}/data/index_data.csv", index_col="DATE")
        stock_data = pd.read_csv(f"./{predict_ticker}/data/stock_data.csv", index_col="Date")
    else:
        os.makedirs(f"{predict_ticker}/data")
        currency_data:pd.DataFrame = web.DataReader(currency_tickers, "fred", start=begin_date, end=datetime.now())
        index_data:pd.DataFrame = web.DataReader(index_tickers, "fred", start=begin_date, end=datetime.now())
        stock_data:pd.DataFrame = historical_stocks_data(stock_tickers)
        currency_data.to_csv(f"./{predict_ticker}/data/currency_data.csv")
        index_data.to_csv(f"./{predict_ticker}/data/index_data.csv")
        stock_data.to_csv(f"./{predict_ticker}/data/stock_data.csv")
    return stock_data, currency_data, index_data

def process_stock_data(stock_tickers:list[str], stock_data:pd.DataFrame, predict_ticker:str) -> pd.DataFrame:
    stock_tickers_copy = stock_tickers.copy()
    stock_tickers_copy.remove(predict_ticker)
    return np.log(stock_data.loc[:, stock_tickers]).diff(RETURN_PERIOD)

def process_index_data(index_data:pd.DataFrame) -> pd.DataFrame:
    return np.log(index_data).diff(RETURN_PERIOD)

def process_currency_data(currency_data:pd.DataFrame) -> pd.DataFrame:
    return np.log(currency_data).diff(RETURN_PERIOD)

def process_dependent_data(stock_data:pd.DataFrame, predict_ticker:str) -> pd.Series:
    y = np.log(stock_data.loc[:, predict_ticker]).diff(RETURN_PERIOD).shift(-RETURN_PERIOD)
    y.name += "_pred"
    return y

def process_endogenous_data(stock_data:pd.DataFrame, predict_ticker:str) -> pd.DataFrame:
    X4 = pd.concat(
        [np.log(stock_data.loc[:, "MSFT"]).diff(i) for i in SELF_BUSINESS_RETURNS_DAYS]
        , axis=1).dropna()
    X4.columns = [f"{predict_ticker}_{i}DR" for i in SELF_BUSINESS_RETURNS_DAYS]
    return X4

def process_data(
        stock_data: pd.DataFrame,
        index_data: pd.DataFrame,
        currency_data: pd.DataFrame,
        stock_tickers: list[str],
        predict_ticker: str
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

def find_k_best_features(k:int, X:pd.DataFrame, y:pd.Series) -> pd.DataFrame:
    best_features = SelectKBest(k=k, score_func=f_regression)
    fit = best_features.fit(X, y)

    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X.columns)

    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = ["Feature", "Score"]
    feature_scores.sort_values(by="Score", ascending=False)

    return feature_scores

def prepare_train_valid_data(
        X:pd.DataFrame,
        y:pd.Series,
        valid_size:float=0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_size = int(len(X) * (1 - valid_size))

    X_train, X_valid = X[:train_size], X[train_size:]
    y_train, y_valid = y[:train_size], y[train_size:]

    return X_train, X_valid, y_train, y_valid

def prepare_models(boosting:bool=True, bagging:bool=True) -> list[tuple[str, BaseEstimator]]:
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

def train_model_list(
        models:list[tuple[str, BaseEstimator]],
        X:pd.DataFrame,
        y:pd.Series,
        train_ARIMA:bool=True,
        train_LSTM:bool=True,
        predict_ticker:str=None
        ) -> tuple[list[str], list[np.ndarray], list[float], list[float]]:
    names = []
    k_fold_results = []
    valid_results = []
    train_results = []

    model_stats_df = pd.DataFrame(columns=["Name", "Mean", "Std", "Train MSE", "Valid MSE"])

    X_train, X_valid, y_train, y_valid = prepare_train_valid_data(X, y)
    for name, model in models:
        names.append(name)

        k_fold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        cv_result = -1 * cross_val_score(model, X_train, y_train, scoring=SCORING, cv=k_fold)
        k_fold_results.append(cv_result)

        trained_model = model.fit(X_train, y_train)
        train_result = mean_squared_error(trained_model.predict(X_train), y_train)
        train_results.append(train_result)

        valid_result = mean_squared_error(trained_model.predict(X_valid), y_valid)
        valid_results.append(valid_result)

        new_df = pd.DataFrame({
            "Name": [name], 
            "Mean": [cv_result.mean()],
            "Std": [cv_result.std()],
            "Train MSE": [train_result],
            "Valid MSE": [valid_result]
        })
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

def plot_K_folds_comparison(names:list[str], k_fold_results:list[np.ndarray], predict_ticker:str) -> None:
    plt.figure(figsize=(15, 8))

    plt.boxplot(k_fold_results)

    plt.title("KFold result")
    plt.xticks(ticks=range(len(names)+1) ,labels=[0]+names)

    plt.savefig(f"{predict_ticker}/{predict_ticker}_k_folds.png")
    # plt.show()

def plot_train_valid_comparison(names:list[str], train_results:list[float], valid_results:list[float], predict_ticker:str) -> None:
    plt.figure(figsize=(15, 8))

    index = np.arange(len(names))
    width = 0.3

    plt.bar(index-width/2, train_results, width=width, label="Train error")
    plt.bar(index+width/2, valid_results, width=width, label="valid error")
    plt.xticks(ticks=range(len(train_results)), labels=names)

    plt.title("Algorithm Comparison")
    plt.legend()

    plt.savefig(f"{predict_ticker}/{predict_ticker}_algo_comparison.png")
    # plt.show()

def prepare_ARIMA_data(
        X:pd.DataFrame, 
        y:pd.Series, 
        predict_ticker:str
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_valid, y_train, y_valid = prepare_train_valid_data(X, y)

    ARIMA_columns = [feature for feature in X_train.columns.to_list()
                     if predict_ticker not in feature]
    X_train_ARIMA = X_train.loc[:, ARIMA_columns]
    X_valid_ARIMA = X_valid.loc[:, ARIMA_columns]
    return X_train_ARIMA, X_valid_ARIMA, y_train, y_valid

def train_ARIMA_model(X:pd.DataFrame, y:pd.Series, predict_ticker:str) -> tuple[ARIMA, float, float]:
    X_train_ARIMA, X_valid_ARIMA, y_train, y_valid = prepare_ARIMA_data(X, y, predict_ticker)

    ARIMA_model = ARIMA(endog=y_train, exog=X_train_ARIMA, order=(1, 0, 0))
    ARIMA_model = ARIMA_model.fit()

    error_ARIMA = mean_squared_error(y_train, ARIMA_model.fittedvalues)
    predict = ARIMA_model.predict(start=len(X_train_ARIMA)-1, end=len(X)-1, exog=X_valid_ARIMA)[1:]
    error_valid_ARIMA = mean_squared_error(y_valid, predict)

    return ARIMA_model, error_ARIMA, error_valid_ARIMA

def prepare_LSTM_data(
        X:pd.DataFrame,
        y:pd.Series,
        seq_len:int=2
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_valid, y_train, y_valid = prepare_train_valid_data(X, y)

    y_train_LSTM, y_valid_LSTM = np.array(y_train)[seq_len-1:], np.array(y_valid)
    X_train_LSTM = np.zeros([X_train.shape[0]+1-seq_len, seq_len, X_train.shape[1]])
    X_valid_LSTM = np.zeros([X_valid.shape[0], seq_len, X_valid.shape[1]])

    for i in range(seq_len):
        X_train_LSTM[:, i, :] = np.array(X_train)[i:X_train.shape[0]+i+1-seq_len, :]
        X_valid_LSTM[:, i , :] = np.array(X)[X_train.shape[0]+i-1:X.shape[0]+i+1-seq_len, :]

    return X_train_LSTM, X_valid_LSTM, y_train_LSTM, y_valid_LSTM

def train_LSTM_model(X:pd.DataFrame, y:pd.Series, plot_history:bool=True) -> tuple[Sequential, float, float]:
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

    history = LSTMmodel.fit(X_train_LSTM, y_train_LSTM,
                            validation_data=(X_valid_LSTM, y_valid_LSTM),
                            epochs=330, batch_size=72, verbose=0, shuffle=False)

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

def find_best_order_ARIMA(
        X:pd.DataFrame,
        y:pd.Series,
        predict_ticker:str,
        p_values:tuple[int]=(0, 1, 2),
        d_values:tuple[int]=(0, 1, 2),
        q_values:tuple[int]=(0, 1, 2)
        ) -> tuple[int]:
    X_train_ARIMA, _, y_train, _ = prepare_ARIMA_data(X, y, predict_ticker)

    def evaluate_ARIMA_model(order: tuple[int, int, int]) -> float:
        ARIMA_model = ARIMA(endog=y_train, exog=X_train_ARIMA, order=order).fit()
        error = mean_squared_error(y_train, ARIMA_model.fittedvalues)
        return error

    best_score, best_order = float("inf"), None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                mse = evaluate_ARIMA_model(order)
                if mse < best_score:
                    best_score, best_order = mse, order
    return best_order

def fine_tune_ARIMA(X:pd.DataFrame, y:pd.Series, predict_ticker:str) -> ARIMA:
    X_train_ARIMA, _, y_train, _ = prepare_ARIMA_data(X, y, predict_ticker)
    ARIMA_fine_tuned = ARIMA(
        endog=y_train,
        exog=X_train_ARIMA,
        order=find_best_order_ARIMA(X, y, predict_ticker)
        ).fit()
    
    return ARIMA_fine_tuned

def plot_test_fine_tuned_ARIMA(X:pd.DataFrame, y:pd.Series, predict_ticker:str) -> None:
    ARIMA_fine_tuned = fine_tune_ARIMA(X, y, predict_ticker)

    X_train_ARIMA, X_valid_ARIMA, y_train, y_valid = prepare_ARIMA_data(X, y, predict_ticker)

    predicted_tuned = ARIMA_fine_tuned.predict(
        start = len(X_train_ARIMA)-1,
        end = len(X)-1,
        exog = X_valid_ARIMA
        )[1:]
    predicted_tuned.index = y_valid.index

    plt.figure()

    plt.plot(np.exp(y_valid).cumprod(), label="actual")
    plt.plot(np.exp(predicted_tuned).cumprod(), label="predict")
    plt.legend()

    plt.savefig(f"{predict_ticker}/{predict_ticker}_backtest.png")
    # plt.show()

def save_model(model:BaseEstimator, predict_ticker:str) -> None:
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
        stock_data, currency_data, index_data = extract_data(stock_tickers, currency_tickers, index_tickers, predict_ticker)

        X, y = process_data(stock_data, index_data, currency_data, stock_tickers, predict_ticker)

        names, k_fold_results, train_results, valid_results = \
            train_model_list(models=prepare_models(), X=X, y=y, predict_ticker=predict_ticker)

        plot_K_folds_comparison(names, k_fold_results, predict_ticker)

        plot_train_valid_comparison(names, train_results, valid_results, predict_ticker)

        plot_test_fine_tuned_ARIMA(X, y, predict_ticker)

        save_model(fine_tune_ARIMA(X, y, predict_ticker), predict_ticker)

