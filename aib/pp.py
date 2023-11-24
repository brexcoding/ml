#_________________- pp.py  ie .  predictions plot _____________--

import pandas as pd 
import matplotlib.pyplot as plt



df = pd.read_csv('predictions')


# Create a line chart
plt.figure(figsize=(10, 6))
plt.plot(df["index"], df["predictions"], marker='o', linestyle='-')
plt.xlabel("Index")
plt.ylabel("Prices")
plt.title("Line Chart of Prices")
plt.grid(True)
plt.show()