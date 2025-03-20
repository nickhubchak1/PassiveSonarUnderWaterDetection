#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------
import numpy as np
from .Layer import Layer

class SoftmaxLayer(Layer):
    #Input: None
    #Output: None

    def __init__(self):
        super().__init__()
        self.dataIn = None

    def forward(self, dataIn):
         #input: dataIn, a (1 by K) matrix
        #output: A (1 by K) matrix

        self.dataIn = dataIn

        self.setPrevIn(dataIn)
        exp_values = np.exp(dataIn - np.max(dataIn))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.setPrevOut(probabilities)
        return probabilities

    def gradient(self):
        #We will worry about this later
        batch_size, input_size = self.dataIn.shape
        softmax_output = self.forward(self.dataIn)

        grad = np.zeros((batch_size, input_size, input_size))

        for i in range(batch_size):
            for j in range(input_size):
                for k in range(input_size):

                    if j ==k:
                        # Diagonal elements: g(z)_j * (1 - g(z)_j)
                        grad[i, j, k] = softmax_output[i, j] * (1 - softmax_output[i, j])
                    
                    else:
                         # Off-diagonal elements: -g(z)_i * g(z)_j
                        grad[i, j, k] = -softmax_output[i, j] * softmax_output[i, k]
        
        return grad
