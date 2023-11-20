#   ****** LR __ linear regression **********
import numpy as np
import pandas as pd
import pyfiglet
from sklearn.linear_model import LinearRegression




# Prepare the new data
new_data = np.array([1.08388 , 1.99965 ,1.48433 ,1.34435])
# Reshape the new data
reshaped_data = new_data.reshape(-1, 1)

print(reshaped_data)

breakpoint()

data = pd.read_csv('mydata')  # load data set
# i went to may close array it was |  ,Close   | and i turned it into -->| index,Close |
#                                  | 0,1.08388 |                      -->|  0,1.08388   | 
# and then droped the index , after i named it in the array                 
data = data.drop('index', axis=1)
data = data[['Close']]


# Create and fit the linear regression model
model = LinearRegression() 
# model training
model.fit(np.arange(len(data)).reshape(-1, 1), data)



text = "predictions "
done = pyfiglet.figlet_format(text ,font='slant')
print(done)


num_predictions = 10

# still dont know how this model pics the data to predict .
# Make predictions 
predictions = model.predict(np.arange(num_predictions).reshape(-1, 1))
print(predictions)

