import pyfiglet 
text = "started "
print( pyfiglet.figlet_format(text ,font='slant' )  )

import numpy as np
import pandas as pd  
from sklearn.linear_model import LinearRegression
import math


data = pd.read_csv('mydata')  # load data set
# converting it to np array
data = data.values
# spliting dataset to numpy arrays
# all rows , coma hhh column one !! compeech! , but this should be 2d thats why [:, :1] instead of [:, 1] 
X = data[:, :1] # and i guess this my input data 
print( "x values ---------> |"  , X)
y = data[:, -1]  # this is my target data 
print( "y values ----------> |"    ,y)




# Create a linear regression model
model = SGDRegressor(learning_rate=0.01, n_iter=100)

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
# Get only the next 10 predictions
Y_pred = Y_pred[-7:] 
print('                          _________________')
print( "         ----------------|  PREDICTIONS  |------------" )
print('                          |_______________|')
print(Y_pred)

import plotly.express as px

# Create a scatter plot
fig = px.scatter(x=X, y=y)

# Add a line plot
fig.add_trace(px.line(x=X, y=Y_pred, color='red'))

# Show the plot
fig.show()
'''
8416 ,141.745,141.6947999999999
8417 ,141.813,141.69634999999988
8418 ,141.833,141.7012499999999
8419 ,141.842,141.7079499999999
8420 ,141.88,141.7186999999999
8421 ,141.937,141.7338999999999
8422, 141.88,141.7466499999999
8423, 141.931,141.7605999999999
8424, 141.918,141.77394999999993
'''