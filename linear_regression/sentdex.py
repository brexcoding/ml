import pandas as pd
import math 
import numpy as np 
from sklearn import preprocessing  , svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from joblib import dump, load

df = pd.read_csv('hourly_EURUSD')

df['hl_pct']  = (df['High'] - df['Close']) /  df['Close'] * 100.0
df['pct_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Close' , 'real_volume' , 'hl_pct' , 'pct_change' , 'tick_volume']]
forecast_col = 'Close'

forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col] . shift(-forecast_out)# labels are values from the df 
# the model use this values to learn -> if your data is starting from 1 to 10 
# and forecast_out is 2 then your label column is from 3 to 10 
#  means it takes 2 values to predict the third

df.dropna(inplace = True)
X = np.array(df.drop(['label'], axis= 1))
X = preprocessing.scale(X)# the shape is (999, 5) ie --> (number of rows,number of columns)

x_previously = X[: - forecast_out] # this means that we will take all the data except the forecast_out 
# which the last 999 scaled values that we are going to input them to the model

# and now we are taking the last 999 values from the x_previously
X_lately =  x_previously[ - forecast_out :]

y = np.array(df['label']) 
# data split
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.2)
# model 
model = LinearRegression(n_jobs = -1)# n_jobs = -1 to use all available CPU cores for parallel computation 
model.fit(X_train , y_train)
pred = model.predict(X_test)
# saving the model 
dump(model, 'model.joblib')

accuracy = model.score(X_test , y_test)


forecast_set = model.predict(X_lately)
print('the forecast set -->', forecast_set ,'accuaracy -->', accuracy ,"forecast out -->", forecast_out)

# processing the real close to plot along with historical data and predictions


##### TODO  I need to make a new file for this data processing
test_close = pd.read_csv('test')
test_close = test_close[['index' , 'Close']]

# added the forecast in the bottom of  the close column for the poting so the predictions come along with the last close prices
df_with_forecast = df._append(pd.DataFrame({
    'Close': forecast_set ,
}), ignore_index=True)

# this is the df and we added the close prices that we fetched later after a wihle
df_with_test_close = df._append(pd.DataFrame({
    'Close': test_close['Close'] ,
}), ignore_index=True)


print(len(df))
plt.style.use('dark_background')
plt.figure(figsize=(10, 5))
plt.plot(df_with_forecast['Close'].head(len(df)), label='close', color='red')
plt.plot(df_with_forecast['Close'].tail(len(forecast_set)), label='forecast', color='green')# ploting the predictions
plt.plot(df_with_test_close['Close'].tail(len(test_close))    , label='last real close', color='black')



plt.xlabel('Time')
plt.ylabel('price')
plt.title('EURUSD Price and Forecast')
plt.legend()
plt.grid(True  ,color='purple')
plt.show()


