#**************LRFS linear regression from scratch "numpy only" *****\
# before i implement linear regression i need to find a linear relationship 
# in the price movement ..like the close price and the sma . when the price goes up 
# the sma goes this i might be able to predict the sma using the close price 
# 
import numpy as np
import pandas as pd

data = pd.read_csv('mydata')
data = data.drop('index', axis=1)  # Droping the index

def loss_function(m , b , data) :# MSE 
    total_error = 0 
    for i in range(len(data)):
        x  = data.iloc[i].Close
        y  = data.iloc[i].sma_20
        total_error += (y - (m*x+b))**2
    total_error / float(len(data))

def gradient_descent(m_now , b_now , data , L):# the optimization algorithm
    m_gradient  = 0
    b_gradient = 0

    n = len(data)
    for i in range(n):
        x  = data.iloc[i].Close
        y  = data.iloc[i].sma_20

        m_gradient += -(2/n) * x * (y - (m_now * x * b_now))
        b_gradient += -(2/n) * (y - (m_now * x * b_now))

    m = m_now - m_gradient * L 
    b = b_now - b_gradient * L 
    return   m , b 

m = 0 
b = 0 
L = 0.00001 # learning rate is good for training
epochs = 150

for i in range(epochs):
    if i % 10 == 0 :
        print(f'======> Epoch :  {i}')
 

    m,b = gradient_descent(m , b , data , L )

print(" ____m___ value is ---->" , m  ) 
print("____b___ value is ---->" , b  )

# Assuming you have new data in a separate DataFrame
new_data = pd.read_csv('mydata')
new_data = new_data.drop('index', axis=1)  # Dropping the index


# Make predictions using the fitted parameters (m and b)
predictions = m * new_data['Close'] + b

print( "the predicted values are --->" , predictions)















