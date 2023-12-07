#   ****** LR __ linear regression **********
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
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.1)
# model 
model = LinearRegression(n_jobs = -1)# n_jobs = -1 to use all available CPU cores for parallel computation 
model.fit(X_train , y_train)
pred = model.predict(X_test)
# saving the model 
dump(model, 'model.joblib')

accuracy = model.score(X_test , y_test)


forecast_set = model.predict(X_lately)
print('the forecast set -->', forecast_set ,'accuaracy -->', accuracy ,"forecast out -->", forecast_out)

# p
# deleting the last close vlaues that match the len of forecast_set ,so i can replace it with forecast_set 
df_without_the_last_values_with_pred_len = df.iloc[:-len(forecast_set)] 
new_df = pd.concat([df_without_the_last_values_with_pred_len, pd.DataFrame(forecast_set, columns=["Close"])], ignore_index=True)




plt.figure(figsize=(10, 5))

plt.plot(df['Close'], label='historical close ', color='green')# ploting the predictions
plt.plot(new_df['Close'], label='Forecast', color='red')


plt.xlabel('Time')
plt.ylabel('Value')
plt.title('EURUSD Price and Forecast')
plt.legend()
plt.grid(True)
plt.show()