# Import numpy and pandas libraries
import numpy as np
import pandas as pd

# Define the activation function (sigmoid)
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Define the derivative of the activation function
def sigmoid_derivative(x):
  return x * (1 - x)

# Define the number of input, hidden and output nodes
input_nodes = 3
hidden_nodes = 4
output_nodes = 2

# Initialize the weights and biases randomly
weights_ih = np.random.rand(input_nodes, hidden_nodes) # weights from input to hidden layer
weights_ho = np.random.rand(hidden_nodes, output_nodes) # weights from hidden to output layer
bias_h = np.random.rand(1, hidden_nodes) # bias for hidden layer
bias_o = np.random.rand(1, output_nodes) # bias for output layer

# Define the learning rate
learning_rate = 0.1

# Define the training data (X) and labels (y)
X = pd.DataFrame([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]) # input data
y = pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]]) # output labels

# Train the neural network for a given number of epochs
epochs = 10000
for epoch in range(epochs):
  
  # Feedforward the input data through the network
  input_layer = X
  hidden_layer = sigmoid(np.dot(input_layer, weights_ih) + bias_h)
  output_layer = sigmoid(np.dot(hidden_layer, weights_ho) + bias_o)

  # Calculate the error between the predicted output and the actual output
  error_output = y - output_layer
  error_hidden = np.dot(error_output, weights_ho.T)

  # Backpropagate the error and update the weights and biases using gradient descent
  delta_output = error_output * sigmoid_derivative(output_layer)
  delta_hidden = error_hidden * sigmoid_derivative(hidden_layer)
  
  weights_ho += learning_rate * np.dot(hidden_layer.T, delta_output)
  weights_ih += learning_rate * np.dot(input_layer.T, delta_hidden)
  
  bias_o += learning_rate * np.sum(delta_output, axis=0)
  bias_h += learning_rate * np.sum(delta_hidden, axis=0)

# Print the final output after training
print("Final output after training:")
print(output_layer.round())
