# _____________ mltb -- machine learning trading bot  ________
import MetaTrader5 as mt5
import  pandas as pd
from datetime import time , datetime 
import time
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
from keras.models import Sequential , load_model
from keras.layers import Dense  ,LSTM
import tensorflow as tf

model = load_model('my_model.h5')
mt5.initialize()
mt5.login(
   login= 66471058   ,                
   password="Khawla1232020.",      
   server="XMGlobal-MT5 2",         
)
symbol = 'EURUSD#'

scaler = MinMaxScaler(feature_range=(0 ,1 ))
data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M3, 0,   9000)
data = pd.DataFrame(data)
data = data.filter(['close'])
dataset = data.values
scaled_data = scaler.fit_transform(dataset)
############

# the last 60 values
input_data = scaled_data[len(scaled_data) - 60: , :]

print('this is  input data .',input_data , \
    'the type is ' , type(input_data) ,\
        'the shape is --->', input_data.shape)


reshaped_input_data = np.reshape(input_data , (1,input_data.shape[0] , 1))
print(reshaped_input_data.shape)


predictions = model.predict(reshaped_input_data)
predictions = scaler.inverse_transform(predictions)
print("the predictions" , predictions)


# balance = mt5.account_info().balance

# Idont no what this function do maybe its not for closing ,, its for 
# checking the orders  but it seems for checking and closing
 
def check_allowed_trading_hours():
    if 7 < datetime.now().hour < 16:
        return True
    else:
        return False
#MY ACCOUNT IS NOT ACCEPTING THE SECOND DIGIT AFTER THE COMMA   
                     
class order():
    
    def buy(symbol):
        point = mt5.symbol_info(symbol).point
        price = mt5.symbol_info_tick(symbol).ask
        deviation = 20
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.1,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": price -60 *point ,
            "tp": price + 110 * point,
            "deviation": deviation,
            "magic": 234000,
            "comment": "machine learning bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # send a trading request
        buy = mt5.order_send(request)
        buy
        print(buy)
    def sell(symbol):
        point = mt5.symbol_info(symbol).point
        price = mt5.symbol_info_tick(symbol).bid
        deviation = 20
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.1 ,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": price +60 *point,
            "tp": price - 110* point,
            "deviation": deviation,
            "magic": 234000,
            "comment": "machine learning bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
         # send a trading request
        sell = mt5.order_send(request)
        sell
        print(sell)






# if __name__ == '__main__':
#     is_initialized = mt5.initialize()
#     print('initialize: ', is_initialized)
#     while True:

        