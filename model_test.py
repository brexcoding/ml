import pandas as pd 
import pickle
import MetaTrader5 as mt5 
import pickle
import xgboost 

# mt5.initialize()
# mt5.login(
#    login= 66471058   ,                
#    password="Khawla1232020.",      
#    server="XMGlobal-MT5 2",         
# )
# symbol = 'AUDCAD#'
# data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M3, 0,   4444)
# data = pd.DataFrame(data)
# data = data.to_csv(symbol)

data = pd.read_csv('AUDCAD#')

pickled_model = pickle.load(open('model0.pkl', 'rb'))
pickled_model.predict(data)