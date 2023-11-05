# ********************** MSE loss function ********************************* 

import numpy as np 

def mse(y_true, y_pred):
  """Calculates the mean squared error between the true and predicted values.

  Args:
    y_true: A NumPy array of true values.
    y_pred: A NumPy array of predicted values.

  Returns:
    A float representing the mean squared error.
  """

  errors = y_true - y_pred
  squared_errors = errors ** 2
  mse = np.mean(squared_errors)
  return mse

y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([0.9, 1.9, 2.8, 3.7, 4.6])

mse = mse(y_true, y_pred)

print(mse)