import  pandas as pd
import numpy as np



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

# Example usage
X = np.array([1, 2, 3, 4, 5]) # this is the input data 
y = np.array([2.5, 3.1, 3.8, 4.5, 5.2])# the target variable

# Fit the linear regression model
m, b = fit_linear_regression(X, y)

# Make predictions for new input data
new_X = np.array([6, 7, 8])
y_pred = predict_linear_regression(new_X, m, b)

print("Predictions:", y_pred)

