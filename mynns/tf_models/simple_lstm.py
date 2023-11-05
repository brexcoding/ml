import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('EURUSD')

# spliting the data to x and y
X = df[['Open', 'High', 'Low', 'Close']]
y = df[['Close']]
# setting my train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


class LSTMModel(tf.keras.Model):
    def __init__(self, units):
        super(LSTMModel, self).__init__()

        self.lstm_layer = LSTMLayer(units=units, return_sequences=True)
        self.dense_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.lstm_layer(inputs)
        x = self.dense_layer(x)

        return x

# Create the model
model = LSTMModel(units=100)

# Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

print('Loss:', loss)
print('Accuracy:', accuracy)
