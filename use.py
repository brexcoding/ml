import pandas as pd 
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense  ,LSTM
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# load model from a HDF5 file
model = tf.keras.models.load_model('cov.h5', compile=False)
# using it to predict the next price
# we need to give the model a new data 
data = pd.read_csv('GBPUSD')
data = data.filter(items = ['Close','Open', 'tick_volume'])
# cnvert the data frame into an array
# get the last 60 close prices

last_60_values = data[-3:].values

print(last_60_values)
breakpoint()
# scale the data to be values between 0 and 1
# defining the scaler
scaler = MinMaxScaler(feature_range=(0 ,1 ))
last_60_close_prices = scaler.fit_transform(last_60_close_prices)
last_60_close_prices_scaled = scaler.transform(last_60_close_prices)
#create an empty list 
X_test = []
# append the past 60 close prices
X_test.append(last_60_close_prices_scaled)
# convert the x_test dataset to a numpy array 
X_test = np.array(X_test)
# reshape the data
X_test = np.reshape(X_test,(X_test.shape[0] , X_test.shape[1], 1 ))
#get the predicted scaled price
pred_price = model.predict(X_test)
# undo scaling 
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)