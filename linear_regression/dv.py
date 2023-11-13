###  dv _  data visulasation 
import plotly.graph_objects as go
import pandas as pd



df = pd.read_csv('mydata')
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

pred_ma20 = [1.07813463 , 1.07813668 ,1.07817787 ,  1.07822111 ,  1.07815522 ,1.07802961,
 1.07801725 , 1.07807285 , 1.07807491 , 1.07800283 , 1.07794723 , 1.07786075 ,  1.07763834]
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



