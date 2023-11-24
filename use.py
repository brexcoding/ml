import pandas as pd 
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential , load_model
from keras.layers import Dense  ,LSTM
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('dark_background')


model = load_model('my_model.h5')
# using it to predict the next price
# we need to give the model a new data 
data = pd.read_csv('EURUSD')

data = data.filter(['Close'])
# convert the dataframe to numpy array
dataset = data.values
# spliting dataset , and math.ceil is just to round the length
train_data_lengh = math.ceil(len(dataset)*.8)
#scale the data
scaler = MinMaxScaler(feature_range=(0 ,1 ))
scaled_data = scaler.fit_transform(dataset)
#################### 


test_data = scaled_data[train_data_lengh - 60: , :]


# create the  datasets x_test and y_test
x_test = []
y_test = dataset[train_data_lengh:,:]
for i in range(60 , len(test_data)):
  x_test.append(test_data[i-60:i,0])


x_test = np.array(x_test)
print('this is X test  ---->',x_test)
print(x_test.shape)

x_test = np.reshape(x_test , (x_test.shape[0], x_test.shape[1] , 1))
print(x_test.shape[1])
print(x_test.shape)

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
print("the predictions" , predictions)
# the loss calculation
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print('the rmse ---->' , rmse)
# ploting the predictions
# ploting the data
train = data[:train_data_lengh]
valid = data[train_data_lengh:]
valid['Predictions'] = predictions
# visualize the data
plt.figure(figsize=(16,8))
plt.title('Model plot')
plt.xlabel('Date',fontsize=18)
plt.ylabel('close price ', fontsize= 18)
plt.plot(train['Close'])
plt.plot(valid[['Close' , 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
