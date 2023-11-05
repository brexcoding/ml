
########## ****** Data Split **********
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('EURUSD')
print(df)
# getting the close prices from the dataframe 
close_price = df['Close']
# creating my new close only dataframe 
close_price_df = pd.DataFrame(close_price)
# saving as a csv file 
data = close_price_df.to_csv('Close')


# spliting the data to x and y
# X = df[['Open', 'High', 'Low', 'Close']]
# y = df[['Close']]
# setting my train and test data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# print(X_train)

# X_train.to_csv('X_train.csv')
# X_test.to_csv("X_test.csv")
# y_train.to_csv("y_train.csv")
# y_test.to_csv('y_test.csv')
# print(' ----------> the splitted data is saved .')