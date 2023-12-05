print('# this code is about preprocessing the data that will be feeded to LSTM  model \
 the X and y sets are scaled  here , the y sets will be unscaled later with scaler.inverse_transform(predictions) function ')
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense  ,LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

data = pd.DataFrame({
  'Close':[91,52,63,74,85,96,77,38 , 92,22],
  'sma_20':[88,50,60,70,80,90,74,33,89,19 ]
})
# data = data.filter(['Close' ]) 
print('this is the dataframe -----------------\n' ,data)
dataset = data.values
train_data_lengh = math.ceil(len(dataset)*.8)

scaler = MinMaxScaler(feature_range=(0 ,1 ))
dataset = scaler.fit_transform(dataset)

train_data = dataset[0:train_data_lengh , : ]
print('this is train data --------------------------\n' , train_data)

# in the for loop we set how many values we input to make predictions based on
x_train = []
y_train = []
for i in range( 4 , len(train_data)):
  x_train.append(train_data[i-4:i ,0])
  y_train.append(train_data[i,0])



print('this is x train ------------------------- \n',x_train)
print('and this is y train ---------------------\n ',y_train) 

#convert the x_train and the y_train from lists to numpy arrays 
x_train , y_train = np.array(x_train) , np.array(y_train)
test_data = dataset[train_data_lengh - 2 : , :]
print('this is the test_data ----------------------- \n',test_data)
x_train = np.reshape(x_train , (x_train.shape[0] , x_train.shape[1] ,1))

# create the test dataset
# create a new array containing scaled values from
test_data = dataset[train_data_lengh - 2 : , :]
# create the  datasets x_test and y_test
x_test = []
y_test = dataset[train_data_lengh:,:]
for i in range(2 , len(test_data)):
  x_test.append(test_data[i-2 :i,0])

print('this is x_test ------------------------------ \n' ,x_test )
print('this is y_test ------------------------------- \n ' , y_test)