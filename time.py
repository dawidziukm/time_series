
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px 
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv('Microsoft_Stock.csv')

#Data Exploration
print(df.head())
print(df.info())
print(df.describe())

#Data Cleaning
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.drop(['Open', 'High', 'Low','Volume'], axis=1,inplace=True)
px.line(df, title='Trend of the time series')

#Normalizing series
df_normalized = df.div(df.iloc[0]).mul(100)
px.line(df_normalized, title='Normalized series')

#Check for Stationarity
df_test = adfuller(df_normalized)
print('Results of Dickey-Fuller test:')
print(f'Test Statistic: {df_test[0]}')
print(f'p-value: {df_test[1]}')
print(f'Critical value: {df_test[4]}')
#p-value is greater than 0.05

df_diff = df_normalized.diff().dropna()
df_test = adfuller(df_diff)
print('Results of Dickey-Fuller test:')
print(f'Test Statistic: {df_test[0]}')
print(f'p-value: {df_test[1]}')
print(f'Critical value: {df_test[4]}')
#p-value is less than 0.05

#Seasonal Decompose
decomp = seasonal_decompose(df_normalized, period=30)
decomp.plot()
plt.show()

decomp = seasonal_decompose(df_diff, period=30)
decomp.plot()
plt.show()

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8))
# ACF plot
plot_acf(df_diff, lags=30, zero=False, ax=ax1)
# PACF plot
plot_pacf(df_diff, lags=30, zero=False, ax=ax2)

#AUTO-ARIMA
#seasonal - monthly
stepwise_model = pm.auto_arima(df_normalized, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())

stepwise_model = pm.auto_arima(df_diff, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())

#seasonal - False
stepwise_model = pm.auto_arima(df_normalized, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=1,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())

#model
data = df_normalized.iloc[:-30]

# Model ARIMA(1,1,0)(2,1,0)[12]
order = (1, 1, 0)  # (p, d, q)
seasonal_order = (2, 1, 0, 12)  # (P, D, Q, s)

model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
results = model.fit(disp=False)

results.plot_diagnostics()
plt.show()

print(results.summary())

forecast_steps = 30 
forecast = results.get_forecast(steps=forecast_steps)

forecast_index = pd.date_range(data.index[-1], periods=forecast_steps + 1, freq='B')[1:]
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()
plt.figure(figsize=(20, 6))
plt.plot(data, label='Historyczne dane')
plt.plot(df_normalized.tail(30), color='b', alpha=0.5, label='Dane testowe')
plt.plot(forecast_index, forecast_mean, color='red', label='Prognoza')
plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='red', alpha=0.1)
plt.legend()
plt.show()


#model 2 - 60days
data = df_normalized.iloc[:-60]

# Model ARIMA(1,1,0)(2,1,0)[12]
order = (1, 1, 0)  # (p, d, q)
seasonal_order = (2, 1, 0, 12)  # (P, D, Q, s)

model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
results = model.fit(disp=False)

results.plot_diagnostics()
plt.show()

print(results.summary())

forecast_steps = 60
forecast = results.get_forecast(steps=forecast_steps)

forecast_index = pd.date_range(data.index[-1], periods=forecast_steps + 1, freq='B')[1:]
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()
plt.figure(figsize=(20, 6))
plt.plot(data, label='Historyczne dane')
plt.plot(df_normalized.tail(60), color='b', alpha=0.5, label='Dane testowe')
plt.plot(forecast_index, forecast_mean, color='red', label='Prognoza')
plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='red', alpha=0.1)
plt.legend()
plt.show()

#model 3 - 90days
data = df_normalized.iloc[:-90]

# Model ARIMA(1,1,0)(2,1,0)[12]
order = (1, 1, 0)  # (p, d, q)
seasonal_order = (2, 1, 0, 12)  # (P, D, Q, s)

model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
results = model.fit(disp=False)

results.plot_diagnostics()
plt.show()

print(results.summary())

forecast_steps = 90 
forecast = results.get_forecast(steps=forecast_steps)

forecast_index = pd.date_range(data.index[-1], periods=forecast_steps + 1, freq='B')[1:]
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()
plt.figure(figsize=(20, 6))
plt.plot(data, label='Historyczne dane')
plt.plot(df_normalized.tail(90), color='b', alpha=0.5, label='Dane testowe')
plt.plot(forecast_index, forecast_mean, color='red', label='Prognoza')
plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='red', alpha=0.1)
plt.legend()
plt.show()