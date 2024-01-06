import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('data/EURUSD_p8.csv')
data["breakout_high"] = data['High'].rolling(3).max().shift(1)
data['breakout_low'] = data['Low'].rolling(3).min().shift(1)
# ploting the data
plot = data[['Close', 'breakout_high' , 'breakout_low']].plot(figsize=(15,8))
plot.set_facecolor('black')  # Set background to black


data['signal']= np.nan
# creating my conditions 
buy = data['Close'] < data['breakout_low']
sell = data['Close'] > data['breakout_low']
# adding the signals to the data
data.loc[buy,'signal'] = 1
data.loc[sell , 'signal'] = -1

open_order = data.loc[data['signal'] == 1].index
close_order = data.loc[data['signal'] == -1].index

plt.figure(figsize=(30, 12))

plt.scatter(open_order, data.loc[open_order]['Close'], color='green', marker='^', label='buy')  # Add label for green scatter
plt.scatter(close_order, data.loc[close_order]['Close'], color='red', marker='v', label='sell')  # Add label for red scatter

plt.plot(data['Close'].index, data['Close'], alpha=0.33, label='EURUSD') # Add label for line plot

plt.legend() # Call legend without explicit labels
plt.show()