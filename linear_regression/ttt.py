# the code bellow is showing how the data is splitted to training and testing sets 
# in the ml model using the train_test_split from  sklearn.model_selection 
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df = pd.DataFrame({
    
    'Close':[91,52,63,74,85,96,77,38]
})
forecast_out = 2
df['label']  = df["Close"] . shift(-forecast_out)
print(df)

# Convert the 'Close' column to a NumPy array
close_array = df['Close'].to_numpy()


X = np.array(df.drop(['label'], axis= 1))
y = np.array(df['label']) 
# X_train , X_test  are features ,  y_train , y_test are labels 
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.2)
print('-------------X train-----------------')
print(X_train)
print('------------- y train ---------------')
print(y_train)
print('------------ x test -------------')
print(X_test)
print('----------------- y test --------------')
print(y_test)


# #  ploting the splited data
# plt.style.use('dark_background')
# plt.figure(figsize=(10, 5))
# plt.scatter(X_train, y_train, label='X_train', color='blue')
# plt.scatter(X_test, y_test, label='X_test', color='red')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.title('Training and Testing Data')
# plt.legend()
# plt.grid(True)
# plt.show()


df = np.array(df.drop(['label'], axis= 1))
x_previously = df[: - forecast_out] # this means that we will take all the data except the forecast_out 
# which the last 999 scaled values that we are going to input them to the model
print("this is x prev , which is an array of features " )
print('-----------------------------')
print( x_previously)
print('-----------------------------')

X_lately =  x_previously[ - forecast_out :]
print('this is the data that we will feed to the model to predict .')

print(X_lately)


