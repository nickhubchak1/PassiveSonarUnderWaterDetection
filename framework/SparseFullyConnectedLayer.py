#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------
from scipy.sparse import random
import numpy as np
from .Layer import Layer

class SparseFullyConnectedLayer(Layer):
    def __init__(self, input_dim, output_dim, density=0.1):
        # Initialize sparse weight matrix (density defines the sparsity level)
        self.weights = random(input_dim, output_dim, density=density, format="csr") * 0.01  # Scaled for Xavier-like initialization
        self.biases = np.zeros((1, output_dim))
    
    def forward(self, X):
        self.input = X
        return X.dot(self.weights) + self.biases

    def backward(self, grad):
        grad_weights = self.input.T.dot(grad)
        grad_biases = np.sum(grad, axis=0, keepdims=True)
        grad_input = grad.dot(self.weights.T)
        return grad_input, grad_weights, grad_biases

    def update_weights(self, grad_weights, grad_biases, learning_rate):
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

    def gradient(self):
    # Placeholder method to satisfy abstract class requirement
        pass
