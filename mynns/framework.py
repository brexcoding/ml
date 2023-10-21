import numpy as np


X = [
    [1 , 2 , 3, 2.5],
    [2.0 , 5.0 , -1.0 , 2.0],
    [-1.5 , 2.7 , 3.3 , -0.8]
]

class Layer_Dense:
    def __init__(self , n_inputs , n_neurons):
        self.weights = 0.10* np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def  forward(self,inputs):
        self.output = np.dot(inputs,self.weights)+self.biases

# making the relue activation function with numpy 
class Relu_Activation:
    def forward(self , inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs , axis=1 , keepdims=True))# to avoid the numbers overflow that comes from exp
        probabilities = exp_values / np.sum(exp_values , axis=1 , keepdims=True)#aka norm values
        self.output = probabilities


activation1 = Relu_Activation()
activation2 = Activation_Softmax()

#  note ---->   Layer_Dense(input size , the number of nurons you want)
layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
activation1.forward(layer1.output)

layer1_outputs = activation1.output

print(layer_outputs)
