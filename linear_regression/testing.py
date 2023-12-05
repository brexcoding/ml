# the code bellow is linear regression with a new method of data split to make the ploting easy
import pyfiglet 
text = "started "
print( pyfiglet.figlet_format(text ,font='slant' )  ) 
import pandas as pd
import math 
import numpy as np 
from sklearn import preprocessing  , svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

df = pd.DataFrame({
  'Close':[91,52,63,74,85,96,77,38]
})
predictions = [1 ,2 ,3 ]

df_without_the_last_values_with_pred_len = df.iloc[:-len(predictions)] 
new_df = pd.concat([df_without_the_last_values_with_pred_len, pd.DataFrame(predictions, columns=["Close"])], ignore_index=True)
print(new_df)




breakpoint()
# df has label column  , dataset has no label column 
df = pd.read_csv('hourly_EURUSD')
df = df[['Open', 'High' , 'Low' , 'Close' , 'tick_volume','real_volume']]
df['hl_pct']  = (df['High'] - df['Close']) /  df['Close'] * 100.0
df['pct_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Close' , 'real_volume' , 'hl_pct' , 'pct_change' , 'tick_volume']]
df.dropna(inplace = True)


train_data_lengh = math.ceil(len(df)*.8)
forecast_out = train_data_lengh
forecast_col = 'Close'

dataset = df 


dataset = preprocessing.scale(dataset)# scaling the data for the train spliting 
train_data = dataset[0:train_data_lengh , : ]




# *******************  train sets *************************

X_train = []
y_train = []
for i in range( 4 , len(train_data)):
  X_train.append(train_data[i-4:i ,0])
  y_train.append(train_data[i,0])

# x_lately = X[: - forecast_out]
# X_lately =  x_lately[ - forecast_out :]

########***********************   testing sets ******************
# since we need the label in testing we will add it to the df
df['label'] = df[forecast_col] . shift(-forecast_out)
testing_dataset = np.array(df['label']) 
print(testing_dataset)

test_data = testing_dataset[train_data_lengh - 2 : , :]
# create the  datasets x_test and y_test
X_test = []
y_test = testing_dataset[train_data_lengh:,:]
for i in range(2 , len(test_data)):
  X_test.append(test_data[i-2 :i,0])


model = LinearRegression(n_jobs = -1)# n_jobs = -1 to use all available CPU cores for parallel computation 
model.fit(X_train , y_train)
pred = model.predict(X_test)

print(pred , 'these are the pedictions')
breakpoint()
forecast_set = model.predict(X_lately)
forecast_set = forecast_set - 0.0230

df = df._append(pd.DataFrame({
    'Close': forecast_set
}), ignore_index=True)

print(forecast_set , accuaracy , forecast_out)

# plt.figure(figsize=(10, 5))
# historical_prices_len = len(df) - len(forecast_set)
# plt.plot(df['Close'].tail(len(forecast_set)), label='forecast', color='green')# ploting the predictions
# plt.plot(df['Close'].head(historical_prices_len), label='Forecast', color='red')


# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.title('EURUSD Price and Forecast')
# plt.legend()
# plt.grid(True)
# plt.show()