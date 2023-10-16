import numpy as np

np.random.seed(0)
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

activation1 = Relu_Activation()


#  note ---->   Layer_Dense(input size , the number of nurons you want)
layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)


layer1.forward(X)
print("data passed threw the layer 1-->",layer1.output)

activation1.forward(layer1.output)
print(activation1.output)