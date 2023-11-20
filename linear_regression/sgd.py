# stochastic gradient decsent 
import pyfiglet 
text = "started "
print( pyfiglet.figlet_format(text ,font='slant' )  ) 

from sklearn.linear_model import SGDRegressor
import numpy as np
import pandas as pd  


# Create a linear regression model
#: The 'learning_rate' parameter of SGDRegressor must be a str among {'constant', 'invscaling', 'adaptive', 'optimal'}
model = SGDRegressor(learning_rate='constant', max_iter=100000)

data = pd.read_csv('mydata')  # load data set
data = data.drop('index', axis=1)  # Droping the index
# converting it to np array
data = data.values

X_train = data[:, :4]# taking all rows of the first 4 columns -> OHLC
print( '-------- X_train ----',X_train)
y_train = data[:, 1] # we want to predict the last column . the sma20
print( 'this is y train ' ,y_train , 'and this is the shape of y train ' , y_train.shape)

# Train the model
model.fit(X_train, y_train)
print('the model is trained , and this is the predicted data  ')
# Make predictions on new data
new_df = pd.read_csv('newdf')
new_df = new_df.drop('index', axis=1) 
new_df = new_df.values
new_data = new_df[-10:, :4]# taking 10 rows of the first 4 columns -> OHLC

y_pred = model.predict(new_data)

print('the shape of y_pred'  ,  y_pred.shape)
print(y_pred)
