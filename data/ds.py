
########## ****** Data Split **********
import pandas as pd
from sklearn.model_selection import train_test_split

import pyfiglet 
text = "data spliting started"
print( pyfiglet.figlet_format(text ,font='basic' )  ) 


df = pd.read_csv('EURUSD')
print(df)

# getting the close prices from the dataframe 
data = df[['Close','Open', 'High', 'Low' , 'sma_20']]
print(data)
# creating my new close only dataframe 
data = pd.DataFrame(data)
# saving as a csv file 
data = data.to_csv('mydata')
T = " ready ..."
done = pyfiglet.figlet_format(T,font='isometric1')
print(done)


breakpoint()
#spliting the data to x and y
X = df[['Open', 'High', 'Low', 'Close']]
y = df[['Close']]
#setting my train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_train)

X_train.to_csv('X_train.csv')
X_test.to_csv("X_test.csv")
y_train.to_csv("y_train.csv")
y_test.to_csv('y_test.csv')
print(' ----------> the splitted data is saved .')