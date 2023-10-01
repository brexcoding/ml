import pandas as pd 
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential , load_model
from keras.layers import Dense  ,LSTM
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
b = breakpoint()


model = load_model('closing_model.keras')


# using it to predict the next price
# we need to give the model a new data 
data = pd.read_csv('GBPUSD')
data = data.filter(items = ['Close','Open', 'tick_volume'])
print(data)
# cnvert the data frame into an array

last_pred_values = data[-60:].values
print(last_pred_values)
# scale the data to be values between 0 and 1
# defining the scaler
scaler = MinMaxScaler(feature_range=(0 ,1 ))
last_pred_values = scaler.fit_transform(last_pred_values)
last_pred_scaled = scaler.transform(last_pred_values)
print('-------->' , last_pred_scaled)

#create an empty list 
X_test = []
# append the past 60 close prices
X_test.append(last_pred_scaled)

# convert the x_test dataset to a numpy array 
