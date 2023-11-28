import pandas as pd
import math 
import numpy as np 
from sklearn import preprocessing  , svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('hourly_EURUSD')
df = df[['Open', 'High' , 'Low' , 'Close' , 'tick_volume','real_volume']]
df['hl_pct']  = (df['High'] - df['Close']) /  df['Close'] * 100.0
df['pct_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Close' , 'real_volume' , 'hl_pct' , 'pct_change' , 'tick_volume']]
forecast_col = 'Close'

forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col] . shift(-forecast_out)
df.dropna(inplace = True)

X = np.array(df.drop(['label'], axis= 1))
X = preprocessing.scale(X)
x_lately = X[: - forecast_out]
X_lately =  x_lately[ - forecast_out :]

y = np.array(df['label']) 
# data split
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.2)
# model 
model = LinearRegression(n_jobs = -1)# n_jobs = -1 to use all available CPU cores for parallel computation 
model.fit(X_train , y_train)
accuaracy = model.score(X_test , y_test)

forecast_set = model.predict(X_lately)

df = df._append(pd.DataFrame({
    'Close': forecast_set
}), ignore_index=True)

print(forecast_set , accuaracy , forecast_out)

plt.figure(figsize=(10, 5))
historical_prices_len = len(df) - len(forecast_set)
plt.plot(df['Close'].tail(len(forecast_set)), label='forecast', color='green')# ploting the predictions
plt.plot(df['Close'].head(historical_prices_len), label='Forecast', color='red')
plt.plot(forecast_set, label='Fast', color='purple')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('EURUSD Price and Forecast')
plt.legend()
plt.grid(True)
plt.show()
