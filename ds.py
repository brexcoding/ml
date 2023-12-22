# ds.py --> i.e , data_split.py
import pandas as pd
import pandas as pd

df = pd.read_csv('EURUSD') 


def split_and_save(df, filename, parts):
  # Calculate the split size
  split_size = int(len(df) / parts)

  # Loop through each part
  for i in range(parts):
    start = i * split_size
    end = (i + 1) * split_size
    
    # Extract the current part
    df_part = df.iloc[start:end]

    # Save the part to a separate CSV file
    df_part.to_csv(f"{filename}_p{i+1}.csv", index=False)

# Read the original CSV file
df = pd.read_csv("EURUSD")

# Split the data into 4 parts
split_and_save(df, "data\\EURUSD", 18)
print('data is splitted and saved .')