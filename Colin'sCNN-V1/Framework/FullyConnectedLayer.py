import numpy as np  
from Framework.Layer import Layer

class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut):
        np.random.seed(42)
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        self.weights = np.random.randn(sizeIn, sizeOut) 
        # Removed biases for CNN functioning
        self.previous_input = None
        self.output = None
    
    def setWeights(self, mtxIn):
        self.weights = mtxIn

    def forward(self, dataIn):
        self.previous_input = dataIn
        output = np.dot(dataIn, self.weights)
        self.output = output
        return output

    def backward(self, gradIn):
        # Ensure grad_input shape matches previous_input shape
        grad_input = np.dot(gradIn, self.weights.T)

        # Calculate gradients for weights
        d_weights = np.dot(self.previous_input.T, gradIn)
        return grad_input, d_weights

    def backward2(self, gradIn):
        # Alternative backward method if needed
        grad_input = np.dot(gradIn, self.weights.T)
        return grad_input

    def compute_gradients_hadamard(self, gradIn):
        d_weights = np.dot(self.previous_input.T, gradIn)
        return d_weights

    def updateWeights(self, d_weights, learning_rate):
        # Adjusting weights with learning rate
        self.weights -= learning_rate * d_weights

    def gradient(self):
        # Placeholder method to satisfy abstract class requirement
        pass
