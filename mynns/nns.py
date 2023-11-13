
# TODO compare it with test.py or lstm.py or firstNN_FS.py
import numpy as np
import pandas as pd

class NeuralNetwork:

    def __init__(self, n_inputs, n_hidden, n_outputs):
        # Initialize weights and biases
        self.w1 = np.random.normal(scale=0.1, size=(n_inputs, n_hidden))
        self.b1 = np.random.normal(scale=0.1, size=(n_hidden,))
        self.w2 = np.random.normal(scale=0.1, size=(n_hidden, n_outputs))
        self.b2 = np.random.normal(scale=0.1, size=(n_outputs,))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, x):
        # Hidden layer activation
        z1 = np.dot(x, self.w1) + self.b1
        a1 = self.sigmoid(z1)

        # Output layer activation
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.sigmoid(z2)

        return a2

def train_neural_network(X, y, epochs, learning_rate):
    # Initialize neural network
    n_inputs = X.shape[1]
    n_hidden = 10 # Change this value to adjust the number of hidden nodes
    n_outputs = y.shape[1]

    model = NeuralNetwork(n_inputs, n_hidden, n_outputs)

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        y_pred = model.feedforward(X)

        # Calculate error
        error = y - y_pred

        # Calculate gradients
        dw2 = np.dot(a2.T, error)
        db2 = np.sum(error, axis=0)

        dz2 = error * y_pred * (1 - y_pred)
        da1 = np.dot(dz2, model.w2.T)

        dw1 = np.dot(X.T, da1)
        db1 = np.sum(da1, axis=0)

        # Update weights and biases
        model.w2 -= learning_rate * dw2
        model.b2 -= learning_rate * db2

        model.w1 -= learning_rate * dw1
        model.b1 -= learning_rate * db1

    return model

data = pd.read_csv('mydata')


X = data['Close'].to_numpy()
y = data['Close'].to_numpy()

print(y.shape)
breakpoint()

# Train the neural network
model = train_neural_network(X, y, epochs=100, learning_rate=0.01)

# Make predictions
X_test = data[['Close']].to_numpy()
y_pred = model.feedforward(X_test)

# Calculate Mean Squared Error (MSE)
predicted_closing_prices = y_pred
mse = np.mean((predicted_closing_prices - data['Close']) ** 2)
print('Mean Squared Error:', mse)
