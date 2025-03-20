#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------
from .Layer import Layer
import numpy as np
import random

class FullyConnectedLayer(Layer):


    #input: sizeIn, the number of features of data coming in 
    #input: sizeOut, the number of features for the data coming out. 
    #output: None
    def __init__(self, sizeIn, sizeOut):
        super().__init__()
        limit = np.sqrt(6 / (sizeIn + sizeOut))  # Xavier/Glorot uniform limit
        self.W = np.random.uniform(-limit, limit, size=(sizeIn, sizeOut))
        self.b = np.zeros((1, sizeOut))  # Biases initialized to zero

    #input: None
    #Output: The (sizeIn by sizeOut) weight matrix.
    def getWeights(self):
        return self.W

    #Input: The (sizeIn by sizeOut) weight matrix
    #Output: None
    def setWeights(self, weights):
        self.W = weights

    #Input: None
    #output: the (1 by sizeOut) bias vector
    def setBiases(self, biases):
        self.b = biases
    
    def getBias(self):
        return self.b
    
    #Input: dataIn, a (1 by D) data matrix
    #Output: A (1 by K) data matrix
    def forward(self, dataIn):

        self.setPrevIn(dataIn)
        #print("self.W size: ", self.W.shape)
        #print("dataIn.size: ", dataIn.shape)
        Y = dataIn@self.W + self.b #The The matmul function is the same simantically as @ intrduced in python 3.5 :)
        #print("FullyConnectedLayer Y.shape: ",Y.shape )
        self.setPrevOut(Y)
        #print("Y shape output of FC layer forward: ", Y.shape)
        return Y
    
    def gradient(self):

        return self.getWeights().T
    
    def backward(self, input_grad):
        #print("input_grad shape: ", input)
        #print("self.gradient() shape: ", self.gradient().shape)
        dG = input_grad @ self.gradient() 
        #     (446,1) x (1, 9)
        #print("dG shape: ", dG.shape)
        return dG

    
    # def backward(self, gradOut):

    #     # Compute the gradient using the previously calculated gradOut (from the next layer)
    #     gradW = np.dot(self.dataIn.T, gradOut)  # (input_size, batch_size)

    #     # Gradient w.r.t. biases: Sum gradients across the batch dimension.
    #     gradB = np.sum(gradOut, axis=0, keepdims=True)  # (1, output_size)

    #     # Gradient w.r.t. input: (batch_size, output_size) x (output_size, input_size) = (batch_size, input_size)
    #     gradIn = np.dot(gradOut, self.W.T)  # (batch_size, input_size)

    #     # Store the gradients for future updates
    #     self.gradW = gradW
    #     self.gradB = gradB
        
    #     return gradIn

    # input eta: is the learning rate
    def updateWeights(self, gradIn, eta):
        N = gradIn.shape[0]
        # if gradIn.shape[1] != self.W.shape[1]:  
        #     gradIn = gradIn.reshape(N, -1) 
        dJdb = np.sum(gradIn, axis = 0) / N
    
        #print(f"dJdb.shape: {dJdb.shape}, self.b.shape: {self.b.shape}")
        dJdW = (self.getPrevIn().T @ gradIn ) / N
        #print(f"dJdW.shape: {dJdW.shape}, self.W.shape: {self.W.shape}")
        
        # print("N :", N)
        #print("self.W shape: ", self.W.shape)
        #print("eta: ", eta)
        #print("dJdW.shape: ", dJdW.shape)
        self.W -= eta * dJdW
        self.b -= eta * dJdb
    
    def backward2(self, input_grad):

        #dW = self.getPrevIn() @ input_grad
        #dB = np.sum(input_grad, axis=0, keepdims=True)
        #print("InputGrad shape: ", input_grad.shape)
        #print("self.gradient shape: ", self.gradient().shape)
        grad_out = input_grad @ self.gradient()
        #self.setWeights(dW)
        #self.setBiases(dB)
        return grad_out

        
    
        