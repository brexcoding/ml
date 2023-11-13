import numpy as np
import pandas as pd 


np.random.seed(0)
# i got to set the training data as  my new X 
X = pd.read_csv('tf_models/Close')
# transformed the dataframe into a numpy array
X = X.values



class Layer_Dense:
    def __init__(self , n_inputs , n_neurons):
        self.weights = 0.10* np.random.randn(n_inputs ,  n_neurons)# By initializing the weights of a neural network with a Gaussian distribution using the randn function, we are more likely to find a good solution to the problem.
        self.biases = np.zeros((1,n_neurons))
    def  forward(self,inputs):
        self.output = np.dot(inputs,self.weights)+self.biases

# making the relue activation function with numpy 
# that takes a numpy array of data and returns a numpy array
class Relu_Activation:
    def forward(self , inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs , axis=1 , keepdims=True))# to avoid the numbers overflow that comes from exp
        probabilities = exp_values / np.sum(exp_values , axis=1 , keepdims=True)#aka norm values
        self.output = probabilities


def mse(y_true, y_pred):
  """Calculates the mean squared error between the true and predicted values.
  Args:
    y_true: A NumPy array of true values.
    y_pred: A NumPy array of predicted values.

  Returns: 
    A float representing the mean squared error.
  """
  errors = y_true - y_pred
  squared_errors = errors ** 2
  mse = np.mean(squared_errors)
  return mse



def train_model(model, X_train, y_train, num_epochs=100, learning_rate=0.01):
  """Trains a neural network model using the Adam optimizer.

  Args:
    model: A neural network model.
    X_train: A NumPy array of training inputs.
    y_train: A NumPy array of training labels.
    num_epochs: The number of training epochs.
    learning_rate: The learning rate.

  Returns:
    The trained model.
  """
  optimizer = AdamOptimizer(learning_rate=learning_rate)

  for epoch in range(num_epochs):
    # Calculate the output of the model and the loss.
    y_pred = model.predict(X_train)
    loss = mse(y_train, y_pred)

    # Calculate the gradients of the loss function with respect to the weights.
    gradients = model.backward(loss)

    # Update the weights of the model using the optimizer.
    updated_gradients = optimizer.update(gradients)
    model.update_weights(updated_gradients)

  return model



activation1 = Relu_Activation()
activation2 = Activation_Softmax()

#  note ---->   Layer_Dense(input size , the number of nurons you want)
# and i can put the **** input.shape[1] **** instead
layer1 = Layer_Dense(X.shape[1],5)
layer2 = Layer_Dense(5,2)

# the first layer with the relue ACTIVATION FUNCTION 
layer1.forward(X)
activation1.forward(layer1.output)
layer1_outputs = activation1.output

# the second layer with the SOFTMAX activation function 
layer2.forward(layer1_outputs)# passing the layer 1 outputs into the second dense layer 
activation2.forward(layer2.output)
layer2_outputs = activation2.output

print( 'layer 2 -----> outputs ',layer2_outputs)
print('the layer 2 shape ------> '   , layer2_outputs.shape)


# Train the model
'''lr = 0.01
num_epochs = 100

for epoch in range(num_epochs):
    w, b = optimizer(w, b, lr, X_train_input, y_train_output)

# Evaluate the model
y_pred = w * X_train_input + b

loss = loss_function(y_pred, y_train_output)

print('Loss:', loss)
'''


# Train the model.
# model = train_model(model, X_train, y_train)

