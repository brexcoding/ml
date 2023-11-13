#   ****** LR __ linear regression **********
import numpy as np
import pandas as pd
import pyfiglet
from sklearn.linear_model import LinearRegression


data = pd.read_csv('mydata')  # load data set
# i went to may close array it was |  ,Close   | and i turned it into -->| index,Close |
#                                  | 0,1.08388 |                      -->|  0,1.08388   | 
# and then droped the index , after i named it in the array                 
data = data.drop('index', axis=1)
data = data.values

# Create and fit the linear regression model
model = LinearRegression(learning_rate=0.01, n_epochs=100) 
# model training
model.fit(np.arange(len(data)).reshape(-1, 1), data)

# Make predictions
#predictions = model.predict(np.arange(len(data)).reshape(-1, 1))

text = "predictions "
done = pyfiglet.figlet_format(text ,font='slant')
print(done)


num_predictions = 10

# Make predictions
predictions = model.predict(np.arange(num_predictions).reshape(-1, 1))
print(predictions)

