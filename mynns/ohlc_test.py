import numpy as np

class Tanh_Activation:
    def forward(self, inputs):
        self.output = np.tanh(inputs)

import numpy as np

def linear_regression(X, y):
    """
    Performs linear regression on the given data.

    Args:
        X: A NumPy array of the independent variables.
        y: A NumPy array of the target variable.

    Returns:
        A NumPy array of the model parameters.
    """
    # Calculate the intercept and slope
    X_transpose = np.transpose(X)
    XTX = np.dot(X_transpose, X)
    Xty = np.dot(X_transpose, y)

    beta = np.linalg.solve(XTX, Xty)

    return beta
# Load the data into NumPy arrays
X = np.loadtxt('data.txt', delimiter=',')
y = np.loadtxt('data.txt', delimiter=',')[:, -1]

# Fit the linear regression model
beta = linear_regression(X, y)

# Print the model parameters
print(beta)

# Define the ReLU activation function
class Relu_Activation:
    def forward(self, inputs):
        """
        Calculates the output of the ReLU activation function.

        Args:
            inputs: A NumPy array of the input data.

        Returns:
            A NumPy array of the output data.
        """
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        """
        Calculates the output of the softmax activation function.

        Args:
            inputs: A NumPy array of the input data.

        Returns:
            A NumPy array of the output data.
        """
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Sigmoid_Activation:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))


class AdamOptimizer:
  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon

    self.m = 0
    self.v = 0

  def update(self, gradients):
    self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
    self.v = self.beta2 * self.v + (1 - self.beta2) * gradients**2

    m_hat = self.m / (1 - self.beta1**0.5)
    v_hat = self.v / (1 - self.beta2**0.5)

    updated_gradients = m_hat / (np.sqrt(v_hat) + self.epsilon)

    return updated_gradients
    
# Define the dense layer
class DenseLayer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        """
        Initializes the dense layer.

        Args:
            input_dim: The number of input features.
            output_dim: The number of output features.
            activation: The activation function to use.
        """
        self.W = np.random.randn(input_dim, output_dim)
        self.b = np.zeros(output_dim)
        self.activation = activation

    def forward(self, X):
        """
        Calculates the output of the dense layer.

        Args:
            X: A NumPy array of the input data.

        Returns:
            A NumPy array of the output data.
        """
        Z = np.dot(X, self.W) + self.b
        if self.activation == 'relu':
            A = Relu_Activation().forward(Z)
        elif self.activation == 'softmax':
            A = Activation_Softmax().forward(Z)
        else:
            raise ValueError('Invalid activation function')
        return A

# Create the dense layer
layer = DenseLayer(input_dim=5, output_dim=1, activation='relu')

# Load the data
data = np.loadtxt('stock_data.csv', delimiter=',')

# Split the data into input and output features
X = data[:, :-1]
y = data[:, -1]

# Train the dense layer
for epoch in range(1000):
    Y_pred = layer.forward(X)
    loss = np.mean((Y_pred - y)**2)
    dW = np.dot(X.T, (Y_pred - y))
    db = np.sum(Y_pred - y)
    layer.W -= dW * 0.01
    layer.b -= db * 0.01

# Evaluate the dense layer
Y_pred = layer.forward(X)
accuracy = np.mean((Y_pred == y).astype(np.float32))
print('Accuracy:', accuracy)
