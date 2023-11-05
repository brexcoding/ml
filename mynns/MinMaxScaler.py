import numpy as np
import pandas as pd

x = np.load('close.npy')
print(x)
print('the type of x is ---> ', type(x))
# data =  pd.read_csv('tf_models/Close')
# print(data)
# x = data.values
# # Save the NumPy array to the file array.npy
# np.save('close.npy', x)
breakpoint()
class MinMaxScaler:
    def __init__(self):
        self.min_values = None
        self.max_values = None

    def fit(self, X):
        self.min_values = np.min(X, axis=0)
        self.max_values = np.max(X, axis=0)

    def transform(self, X):
        return (X - self.min_values) / (self.max_values - self.min_values)

# Create a MinMaxScaler
scaler = MinMaxScaler()
# Fit the scaler to the training data
scaler.fit(X_train)
# Transform the training data
X_train_scaled = scaler.transform(X_train)
# Transform the test data
X_test_scaled = scaler.transform(X_test)