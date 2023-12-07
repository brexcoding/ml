# ds.py --> i.e , data_split.py
import pandas as pd
import pandas as pd

df = pd.read_csv('EURUSD') 

# splitting the df in 4 as a spliting start
first_split = int(len(df)/4)

df_part1 = df.iloc[:first_split]
df_part2 = df.iloc[first_split: first_split*2]
df_part3 = df.iloc[first_split*2 : first_split*3]
df_part4 = df.iloc[first_split*3 : ]

df_part1.to_csv('EURUSD_p1')
df_part2.to_csv('EURUSD_p2')
df_part3.to_csv('EURUSD_p3')
df_part4.to_csv('EURUSD_p4')