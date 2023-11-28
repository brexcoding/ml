# ______ vcr ,ie Volume and close relation ____________
# the goal of this code is to check if there is a linear relation between the 
# volume  and the close
import pandas as pd 

df = pd.read_csv('hourly_EURUSD')
# we want to see if high volume means high close 
# open + prev_volume + prev_close  = next_close