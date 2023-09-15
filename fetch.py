
# import pytz 
import pandas as pd
# import talib as ta
import pandas as pd
# import talib as ta
import MetaTrader5  as mt5
from  datetime import datetime 

mt5.initialize()
mt5.login(
   login= 66471058   ,                
   password="Khawla1232020.",      
   server="XMGlobal-MT5 2",         
)
symbol = 'GBPUSD'
data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M3, 0,   90000)
data = pd.DataFrame(data)


data = data.rename(columns={"time": "Time", "open": "Open" , "high": "High",
    "low":"Low" , 'close':'Close'})

data = data.to_csv(symbol)
print("done fetching historical data for the symbol--->",symbol)



data['sma_20'] = ta.SMA(data['Close'], timeperiod=20)
data['sma_200'] = ta.SMA(data['Close'], timeperiod=200)

breakpoint()


symbols = ("EURUSD" ,"AUDCAD", "USDJPY") # ,"USDJPY",'EURJPY',  ,"CADJPY" ,"USDJPY" ,"AUDCAD","USDCAD","USDCHF")

for symbol in symbols :
    # timezone = pytz.timezone("Etc/UTC")
    # # create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
    # utc_from = datetime(2023, 7 , 19 , tzinfo=timezone)
    # # get 10 EURUSD H4 bars starting from 01.10.2020 in UTC time zone
    # data = mt5.copy_rates_from(symbol , mt5.TIMEFRAME_M5, utc_from,  300)
    data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M3, 0,   4444)
    data = pd.DataFrame(data)
    
    data = data.rename(columns={"time": "Time", "open": "Open" , "high": "High",
     "low":"Low" , 'close':'Close'})
    
    
    data["Time"]= pd.to_datetime(data['Time'], unit = "s")
    hourlydata = data['Time'].dt.hour
    session = (hourlydata >= 6) & (hourlydata <= 17)
    data['session'] = session
    # definging the mas and 
    data = data.to_csv(symbol)
    print("done fetching historical data for the symbol--->",symbol)


#>>>>>>>>>>>>>>>>>>>>>>>> adding signals section >>>>>>>>>>>>>>>>>>>>>>
for symbol in symbols : 
    
    data =pd.read_csv(symbol)
    previous_2nd_bar_open = data.iloc[-2].Open
    previous_2nd_bar_close = data.iloc[-2].Close
    ma_200 = ta.MA(data['Close'], timeperiod=200 , matype=0)
    ma_20 = ta.MA(data['Close'], timeperiod=20 , matype=0)
    data['ema_20']= ta.EMA(data['Close'], timeperiod=20 )
    data['sma_20'] = ta.SMA(data['Close'], timeperiod=20)
    data['sma_200'] = ta.SMA(data['Close'], timeperiod=200)
    
###### bullish section ##########################################################

    # adding to the dataframe as a column
    data['BullishCross'] =  ma_20 >ma_200
    print("adding bullish cross")
    #data['morningstar'] = ta.CDLMORNINGDOJISTAR(data['Open'] ,data['High'],data['Low'],data['Close'], penetration=0) 
    #print('adding morningstar pattern detection')
    data['hammer'] =  ta.CDLHAMMER(data['Open'] ,data['High'],data['Low'],data['Close'])
    print('adding hammer pattern detection')
    data['ma_20'] = ta.MA(data['Close'], timeperiod=20 , matype=0)
    print('adding ma20 indicator')
    data['piercingpattern'] = ta.CDLPIERCING(data['Open'] ,data['High'],data['Low'],data['Close'])
    print('adding piercing pattern detection')
    data['threesoliders']= ta.CDL3STARSINSOUTH(data['Open'] ,data['High'],data['Low'],data['Close'])
################# mixed sign section >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  
    data['engulfing'] = ta.CDLENGULFING(data['Open'] ,data['High'],data['Low'],data['Close']) # if its == -100  that means bearish engulfing
    print('adding engulfing pattern detection')
    data['three_inside'] = ta.CDL3INSIDE(data['Open'] ,data['High'],data['Low'],data['Close']) # --------->new
    print('adding three_inside pattern detection')
    data['three_outside'] = ta.CDL3OUTSIDE(data['Open'] ,data['High'],data['Low'],data['Close']) 
    print('adding three_outside pattern detection')
    data['three_line_strike'] = ta.CDL3LINESTRIKE(data['Open'] ,data['High'],data['Low'],data['Close']) 
    print('adding   three_line_strike pattern detection')
    data['sticksandwish'] = ta.CDLSTICKSANDWICH(data['Open'] ,data['High'],data['Low'],data['Close']) 
    print('adding sticksandwish pattern detection')
############ bearish signals ######################################################
   
    # adding as a column 
    data['darkcloudcover'] = ta.CDLDARKCLOUDCOVER(data['Open'] ,data['High'],data['Low'],data['Close'], penetration=0) 
    print('adding darkcloudcover pattern recognition')
    data['BearishCross'] =  ma_20<ma_200
    print('adding the bearish cross signal')
    data['eveningstar']  =  ta.CDLEVENINGSTAR(data['Open'] ,data['High'],data['Low'],data['Close'], penetration=0) 
    print('adding evening star pattern recognition')
    data['eveningdojistar'] = ta.CDLEVENINGDOJISTAR(data['Open'] ,data['High'],data['Low'],data['Close'], penetration=0)
    print('adding evening doji pattern recognition')
    data['gravestonedoji'] =   ta.CDLGRAVESTONEDOJI(data['Open'] ,data['High'],data['Low'],data['Close'])# gives around the 50% with the bullish cross  
    print('adding the gravestonedoji pattern recognition')
    data['invertedhammer'] =  ta.CDLINVERTEDHAMMER(data['Open'] ,data['High'],data['Low'],data['Close'])
    print("adding invertedhammer pattern recognition  ")
    data['hangingman']= ta.CDLINVERTEDHAMMER(data['Open'] ,data['High'],data['Low'],data['Close'])
    print('adding the hangingman pattern recognition')
    data.to_csv(symbol)
    print('converting the fetched data into a csv and saving it')
    print("done adding patterns sir !! ----> ",symbol)