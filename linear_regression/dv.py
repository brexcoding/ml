###  dv _  data visulasation 
import plotly.graph_objects as go
import pandas as pd
import numpy as np


df = pd.read_csv('hourly_EURUSD')
# Convert the index column to datetime format
newdf = pd.read_csv('newdf')

# Create a Plotly figure
fig = go.Figure()

# Add the OHLC candlesticks
fig.add_trace(go.Candlestick(x=df['index'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'))
# Add the moving averages
fig.add_trace(go.Scatter(x=df['index'], y=df['sma_20'], name='SMA20', line=dict(color='orange', width=1)))
# adding the actual vales of the sma ,to compare them with the prdicted values
fig.add_trace(go.Scatter(x=newdf['index'], y=newdf['sma_20'], name='actual-SMA20', line=dict(color='green', width=1)))
# adding the volume plot 
fig.add_trace(go.Scatter(x=df['index'], y=df['tick_volume'], name='volume', line=dict(color='blue', width=1)))

pred_ma20 = [1.47138141 ,1.47129299 ,1.47126805 ,1.47115708 ,1.47114797, 1.47123179
, 1.47136779 , 1.47144253 ,1.47159435 ,1.47164649]
pred_ma20 = np.array(pred_ma20)
# adding the actual vales of the sma ,to compare them with the prdicted values
fig.add_trace(go.Scatter(x=newdf['index'], y= pred_ma20  , name='PRED-SMA20', line=dict(color='yellow', width=1)))
# Set the layout of the figure
fig.update_layout(
    title='OHLC Data and Moving Averages',
    xaxis_title='Date',
    yaxis_title='Price',
    legend_title='Traces'
)

# Show the figure
fig.show()



