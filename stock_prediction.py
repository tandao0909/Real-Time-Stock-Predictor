#!/usr/bin/env python
# coding: utf-8

# # Stock Return Prediction

# ## Problem Definition
# 
# In this supervised regression framework, we will predict the weekly return of Microsoft (MSFT). We want to understand what affects the Microsoft stock price.
# 
# We will use the stock price of its competitors (which are other big tech companies), the indices associates with USA and tech sector, and some currency exchange rate involved USD. In particular, we will use these independent variables:
# - Stocks: Amazon (AMZN), Google (GOOGL), IBM (IBM), Apple (AAPL)
# - Currency: USD/JPY and GBP/USD
# - Indices: S&P 500, Dow Jones, Nasdaq 100

# ## Loading the data and required packages

# ### Loading the packages

# For supervised regression models

# In[48]:


from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor


# For data analysis and model evaluation

# In[49]:


from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import chi2, f_regression, SelectKBest


# For deep learning models

# In[77]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer
from tensorflow.keras.optimizers import SGD
from scikeras.wrappers import KerasRegressor


# For time series models

# In[51]:


from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm


# For data preparation and visualization

# In[52]:


import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf
from yahoo import historical_stocks_data


# ### Loading the data

# Now, we will extract the data using the `pandas_datareader` library. 

# In[53]:


stock_tickers = ["MSFT", "GOOGL", "IBM"]
currency_tickers = ["DEXJPUS", "DEXUSUK"]
index_tickers = ["SP500", "DJIA", "NASDAQ100"]

from datetime import datetime
from dateutil.relativedelta import relativedelta
begin_date = datetime.now() - relativedelta(years=10, days=-1)

# Uncomment the 6 lines below when run the first time
# currency_data:pd.DataFrame = web.DataReader(currency_tickers, "fred", start=begin_date, end=datetime.now())
# index_data:pd.DataFrame = web.DataReader(index_tickers, "fred", start=begin_date, end=datetime.now())
# stock_data:pd.DataFrame = historical_stocks_data(stock_tickers)
# currency_data.to_csv("./data/currency_data.csv")
# index_data.to_csv("./data/index_data.csv")
# stock_data.to_csv("./data/stock_data.csv")

currency_data = pd.read_csv("./data/currency_data.csv", index_col="DATE")
index_data = pd.read_csv("./data/index_data.csv", index_col="DATE")
stock_data = pd.read_csv("./data/stock_data.csv", index_col="Date")


# Next, we need to choose a time frame to predict. We will predict weekly returns. Hence, we would use 5 business days (stock exchange is stopped in Saturday and Sunday).

# In[54]:


return_period = 5

SELF_BUSINESS_RETURNS_DAYS = [5, 15, 30, 60]


# We now define the independent variables and dependent variable:
# 
# Y: 
# 
# MSFT Future Returns (in the next 5 days)
# 
# X:
# ```
#     GOOGL 5 Business Day Returns
#     ISM 5 Business Day Returns
#     USD/JPY 5 Business Day Returns
#     GBP/USD 5 Business Day Returns
#     S&P 500 5 Business Day Returns
#     DOW JONES 5 Business Day Returns
#     MSFT 5 Business Day Returns
#     MSFT 15 Business Day Returns
#     MSFT 30 Business Day Returns
#     MSFT 60 Business Day Returns
# ```

# In[55]:


stock_data


# In[56]:


y = np.log(stock_data.loc[:, "MSFT"]).diff(return_period).shift(-return_period)
y.name += "_pred"
# y = pd.concat([pd.to_datetime(stock_data["Date"]), y], axis=1)
# y.set_index("Date", inplace=True)
y


# In[57]:


X1 = np.log(stock_data.loc[:, ("GOOGL", "IBM")]).diff(return_period)
X2 = np.log(currency_data).diff(return_period)
X3 = np.log(index_data).diff(return_period)

X4 = pd.concat(
    [np.log(stock_data.loc[:, "MSFT"]).diff(i) for i in SELF_BUSINESS_RETURNS_DAYS]
    , axis=1).dropna()
X4.columns = [f"MSFT_{i}DR" for i in SELF_BUSINESS_RETURNS_DAYS]

X = pd.concat([X1, X2, X3, X4], axis=1).dropna()
X.head(10)


# In[58]:


y


# In[59]:


dataset = pd.concat([X, y], axis=1).dropna().iloc[::return_period, :]
X = dataset.loc[:, X.columns]
y = dataset.loc[:, y.name]
dataset


# # Data Visualization

# In[60]:


dataset.hist(bins=50, sharex=True, sharey=True, figsize=(12, 12))
plt.show()


# In[61]:


correlation = dataset.corr()
plt.figure(figsize=(16, 16))
plt.title("Correlation matrix")
sns.heatmap(correlation, square=True, annot=True, vmax=1)


# In[62]:


plt.figure(figsize=(16, 16))
scatter_matrix(dataset, figsize=(12, 12))
plt.show()


# # Time series analysis

# In[63]:


decomposition = sm.tsa.seasonal_decompose(y, period=52)
fig = decomposition.plot()
fig.set_figheight(8)
fig.set_figwidth(15)
plt.show()


# # Feature Selection

# In[64]:


best_features = SelectKBest(k=5, score_func=f_regression)
fit = best_features.fit(X, y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
feature_scores = pd.concat([df_columns, df_scores], axis=1)
feature_scores.columns = ["Feature", "Score"]
feature_scores.sort_values(by="Score", ascending=False)


# # Train/Test split and evaluation metric

# In[65]:


valid_size = 0.2

train_size = int(len(X) * (1 - valid_size))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# In[66]:


num_folds = 10
seed = 42
scoring = "neg_mean_squared_error"


# # Compare models and algorithms

# In[67]:


models = []
models.append(("Linear", LinearRegression()))
models.append(("Lasso", Lasso()))
models.append(("Elastic", ElasticNet()))
models.append(("KNN", KNeighborsRegressor()))
models.append(("CART", DecisionTreeRegressor()))
models.append(("SVR", SVR()))
models.append(("MLP", MLPRegressor()))
models.append(("ABR", AdaBoostRegressor()))
models.append(("GBR", GradientBoostingRegressor()))
models.append(("RFR", RandomForestRegressor()))
models.append(("ETR", ExtraTreesRegressor()))


# In[68]:


names = []
k_fold_results = []
test_results = []
train_results = []
for name, model in models:
    names.append(name)

    k_fold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    cv_result = -1*cross_val_score(model, X_train, y_train, scoring=scoring, cv=k_fold)
    
    k_fold_results.append(cv_result)

    trained_model = model.fit(X_train, y_train)
    train_result = mean_squared_error(trained_model.predict(X_train), y_train)
    train_results.append(train_result)

    test_result = mean_squared_error(trained_model.predict(X_test), y_test)
    test_results.append(test_result)

    print(f"{name}: Mean: {cv_result.mean()} Std: {cv_result.std()} Train MSE: {train_result}, Test MSE: {test_result}")


# In[69]:


plt.figure(figsize=(15, 8))
plt.title("KFold result")
plt.boxplot(k_fold_results)
plt.xticks(ticks=range(len(names)+1) ,labels=[0]+names)
plt.show()


# In[70]:


plt.figure(figsize=(15, 8))

index = np.arange(len(names))
width = 0.3

plt.title("Algorithm Comparison")
plt.bar(index-width/2, train_results, width=width, label="Train error")
plt.bar(index+width/2, test_results, width=width, label="Test error")

plt.xticks(ticks=range(len(train_results)), labels=names)
plt.legend()
plt.show()


# # Time Series Based models: ARIMA and LSTM

# In[71]:


ARIMA_columns = [feature for feature in X.columns.to_list() if "MSFT" not in feature]
X_train_ARIMA = X_train.loc[:, ARIMA_columns]
X_test_ARIMA = X_test.loc[:, ARIMA_columns]


# In[72]:


ARIMA_model = ARIMA(endog=y_train, exog=X_train_ARIMA, order=(1, 0, 0))
ARIMA_model = ARIMA_model.fit()


# In[73]:


error_ARIMA = mean_squared_error(y_train, ARIMA_model.predict(exog=X_train_ARIMA))
predict = ARIMA_model.predict(start=len(X_train_ARIMA)-1, end=len(X)-1, exog=X_test_ARIMA)[1:]
error_test_ARIMA = mean_squared_error(y_test, predict)


# # LSTM model

# We are dealing with time series model. In this notebook, `seq_len` means the length of the whole sequence, which means the length of the predict sequence of days and the day we forecast. Here we use a day to forecast the next day.

# In[74]:


seq_len = 2

y_train_LSTM, y_test_LSTM = np.array(y_train)[seq_len-1:], np.array(y_test)
X_train_LSTM = np.zeros([X_train.shape[0]+1-seq_len, seq_len, X_train.shape[1]])
X_test_LSTM = np.zeros([X_test.shape[0], seq_len, X_test.shape[1]])
for i in range(seq_len):
    X_train_LSTM[:, i, :] = np.array(X_train)[i:X_train.shape[0]+i+1-seq_len, :]
    X_test_LSTM[:, i , :] = np.array(X)[X_train.shape[0]+i-1:X.shape[0]+i+1-seq_len, :]


# In[81]:


def create_LSTMmodel(neurons=12, learn_rate=0.01, momentum=0) -> Sequential:
    model = Sequential(
        [   
            InputLayer([X_train_LSTM.shape[1], X_train_LSTM.shape[2]]),
            LSTM(50),
            Dense(1),
        ]
    )
    model.compile(loss="mse", optimizer="adam")
    return model
LSTMmodel = create_LSTMmodel(12, learn_rate=0.01, momentum=0)
history = LSTMmodel.fit(X_train_LSTM, y_train_LSTM, validation_data=(X_test_LSTM, y_test_LSTM),epochs=330, batch_size=72, verbose=0, shuffle=False)


# In[82]:


plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend()
plt.show()


# In[84]:


error_LSTM = mean_squared_error(y_train_LSTM, LSTMmodel.predict(X_train_LSTM))
error_test_LSTM = mean_squared_error(y_test_LSTM, LSTMmodel.predict(X_test_LSTM))


# In[86]:


test_results.append(error_test_ARIMA)
test_results.append(error_test_LSTM)

train_results.append(error_ARIMA)
train_results.append(error_LSTM)

names.append("ARIMA")
names.append("LSTM")


# # Comparison of all the algorithms (including Time Series Algorithms)

# In[87]:


plt.figure(figsize=(15, 8))

index = np.arange(len(names))
width = 0.3

plt.title("Algorithm Comparison")
plt.bar(index-width/2, train_results, width=width, label="Train error")
plt.bar(index+width/2, test_results, width=width, label="Test error")

plt.xticks(ticks=range(len(train_results)), labels=names)
plt.legend()
plt.show()


# # Fine tune the model using grid search

# In[91]:


def evaluate_ARIMA_model(order: tuple[int]) -> float:
    ARIMA_model = ARIMA(endog=y_train, exog=X_train_ARIMA, order=order)
    ARIMA_model.fit()
    error = mean_squared_error(y_train, ARIMA_model.predict(X_train_ARIMA))
    return error

def evaluate_model(p_values, d_values, q_values) -> tuple[int]:
    best_score, best_order = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_ARIMA_model(order)
                    if mse < best_score:
                        best_score, best_order = mse, order
                except:
                    continue
    print(f"Best order {order}")

import warnings
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1]
warnings.filterwarnings("ignore")
evaluate_model(p_values, d_values, q_values)


# # Test the model

# In[92]:


# prepare model
model_fit_tuned = ARIMA(endog=y_train, exog=X_train_ARIMA, order=[2, 1, 1]).fit()

# estimate accuracy on validation set
predicted_tuned = model_fit_tuned.predict(start = len(X_train) -1 ,end = len(X) -1, exog = X_test_ARIMA)[1:]
print(mean_squared_error(y_test,predicted_tuned))


# After tuning the model and picking the best ARIMA model or the order 2,0 and 1 we select this model and can it can be used for the modeling purpose. 

# In[96]:


predicted_tuned.index = y_test.index
plt.plot(np.exp(y_test).cumprod(), label="actual")
plt.plot(np.exp(predicted_tuned).cumprod(), label="predict")
plt.legend()


# ### Summary

# In[ ]:




