#_________________- pp.py  ie .  predictions plot _____________--

import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use('dark_background')



df = pd.read_csv('predictions')
data = pd.read_csv('hourlydata')


# Create a line chart
plt.figure(figsize=(10, 6))
plt.plot(data["index"], data["close"], marker='o', linestyle='-' ,  color='red')
plt.plot(df["index"], df["predictions"], marker='o', linestyle='-' ,  color='green')
plt.xlabel("Index")
plt.ylabel("Prices")
plt.title("Line Chart of predictions")
plt.show()