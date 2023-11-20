# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense  ,LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

data = pd.read_csv('EURUSD')
data.shape

#Cretung a new dataframe with only the close
data = data.filter(['Close'])
# convert the dataframe to numpy array
dataset = data.values
# spliting dataset
train_data_lengh = math.ceil(len(dataset)*.8)
train_data_lengh

#scale the data 
scaler = MinMaxScaler(feature_range=(0 ,1 ))
scaled_data = scaler.fit_transform(dataset)
scaled_data

train_data = scaled_data[0:train_data_lengh , : ]
x_train = []
y_train = []
for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i ,0])
  y_train.append(train_data[i,0])
  if i<= 60:
    print(x_train)
    print(y_train)
    print()

#convert the x_train ansd the y_train to numpy arrays
x_train , y_train = np.array(x_train) , np.array(y_train)

#reshape the data ,,,,,,,,, convert it to 3d
x_train = np.reshape(x_train , (x_train.shape[0] , x_train.shape[1] ,1))
x_train.shape

# build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True , input_shape = (x_train.shape[1] , 1)))
model.add(LSTM(50 , return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#compile the model
model.compile(optimizer='adam' , loss='mse')
# import keras
# model.compile(optimizer=keras.optimizers.Adam(), loss="mse")

# train the model
model.fit(x_train , y_train , batch_size=1 , epochs=1)

# create the test dataset
# create a new array containgn scaled values from
test_data = scaled_data[train_data_lengh - 60: , :]
# create the  datasets x_test and y_test
x_test = []
y_test = dataset[train_data_lengh:,:]
for i in range(60 , len(test_data)):
  x_test.append(test_data[i-60:i,0])

# converting it into a numpy aray ,,,,lstm does not accept 2d ,so we conver it to 3d
x_test = np.array(x_test)
# reshaping it
x_test = np.reshape(x_test , (x_test.shape[0], x_test.shape[1] , 1))

# using the model to predict
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# evaluating the model .wuth the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse

# ploting the data
train = data[:train_data_lengh]
valid = data[train_data_lengh:]
valid['Predictions'] = predictions
# visualize the data
plt.figure(figsize=(9,4.5))
plt.title('Model plot')
plt.xlabel('Date',fontsize=18)
plt.ylabel('close price ', fontsize= 18)
plt.plot(train['Close'])
plt.plot(valid[['Close' , 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# showing the predicted values
valid

# using it to predict the next price