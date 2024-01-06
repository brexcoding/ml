import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential , load_model
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


# Load OHLC data
data = pd.read_csv('data\\EURUSD_p8.csv')
features = data[['Open', 'High', 'Low', 'volume','spread', 'Close']]

# Extract OHLC data
# # Normalize OHLC data
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(features)

# Create sequences
sequences = []
for i in range(60, len(normalized_data)):
    sequences.append(normalized_data[i - 60:i])

# Convert sequences to numpy array
sequences = np.array(sequences)
# selecting the new_data which is the first 60 sample
new_data = sequences[0 , :, :] 

#reshaping it to 3d to feed it to the model
new_data =  np.expand_dims(new_data, axis=0)


# loading the model 
model = load_model('the8features_model12.h5')

predictions = model.predict(new_data)
print(predictions)

predictions = scaler.inverse_transform(predictions.reshape(-1,6))
# getting only the close predictions which is the last column 
predictions= predictions[: , -1]


forecast_set =predictions
historical_close = data.filter(['Close'])
first_60_closes = historical_close.head(60)
# first 60 closes + the predictions
new_df = pd.concat([first_60_closes, pd.DataFrame(forecast_set, columns=["Close"])], ignore_index=True)
print(new_df)


plt.style.use('dark_background')
plt.figure(figsize=(10, 5))
plt.plot(new_df['Close'], label='Forecast', color='red')
plt.plot(historical_close, label='historical close ', color='green')# ploting the predictions



plt.xlabel('Time')
plt.ylabel('Value')
plt.title('EURUSD Price and Forecast')
plt.legend()
plt.grid(True)
plt.show()

# the way how the data  is preprocessed to be fed to the model
'''
normalized_data = pd.DataFrame({
    'Close':[91,52,63,74,85,96,77,38 ],
    'open': [3 ,6 ,6 ,6 ,7,9,99 ,91  ] ,
    'sma20':[4 ,5,7 , 8, 88, 88, 66,77]
})

sequences = []
for i in range(3, len(normalized_data)):
    sequences.append(normalized_data[i - 3:i])

# Convert sequences to numpy array
sequences = np.array(sequences)
y_train = sequences[:, :, -1:] 
print('this is y train \n ',y_train)

print(sequences)
new_data = sequences[0, :, :] 
print('this is the new data')
print(new_data)

'''