import  pandas as pd
import numpy as np

import pyfiglet 
text = "slr "
print( pyfiglet.figlet_format(text ,font='isometric1' )  ) 

# the bellow code is AI generated and it need to be checked later , bcs i think it lacks trainging funcion like gradient decsent
def fit_linear_regression(X, y):
    """Fits a linear regression model to the data."""

    # Calculate the slope and intercept
    m = np.sum((y - np.mean(y)) * (X - np.mean(X))) / np.sum((X - np.mean(X)) ** 2)
    b = np.mean(y) - m * np.mean(X)

    return m, b

def predict_linear_regression(X, m, b):
    """Predicts the output values for the given input data using the fitted linear regression model."""

    y_pred = m * X + b
    return y_pred

data = pd.read_csv('mydata')  # load data set
data = data.drop('index', axis=1)  # Droping the index
# converting it to np array
data = data.values

X = data[:, :4]# taking all rows of the first 4 columns -> OHLC

y = data[:, -1]

print('trainging the model ...')
m, b = fit_linear_regression(X, y)
print('done ... ')

print('--------------- predictions -------------------------')
new_df = pd.read_csv('newdf')
new_df = new_df.drop('index', axis=1) 
new_df = new_df.values
new_data = new_df[:, :4]

y_pred = predict_linear_regression(new_X, m, b)

print("Predictions:", y_pred)

