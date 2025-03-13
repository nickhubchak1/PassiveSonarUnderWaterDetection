#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------
import numpy as np
from .Layer import Layer

class LogisticSigmoidLayer(Layer):
    #Input: None
    #Output: None

    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        #input: dataIn, a (1 by K) matrix
        #output: A (1 by K) matrix
        self.setPrevIn(dataIn)
        Y = 1 / (1+np.exp(-dataIn))
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        
        #sigmoid = 1/ (1+np.exp(-self.dataIn))
        sigmoid = self.forward(self.getPrevIn())
        batch_size, input_size = self.getPrevIn().shape

        grad = np.zeros((batch_size, input_size, input_size))

        for i in range(batch_size):
            diag = sigmoid[i] * (1-sigmoid[i])

            #NOTE: Be careful when using numpy’s 𝑑iag
            # function. Using it on a 1D matrix vs vector
            # yields different results. You’ll likely want to
            # pass it a vector, but make sure to test that
            # it’s providing what you want.
            grad[i] = np.diag(diag) #Fills diagonal of jacobian
        return grad
    
    def backward(self, gradIn):
        sigmoid_grad = self.getPrevOut() * (1 - self.getPrevOut())
        
        # Element-wise multiply the incoming gradient with the sigmoid derivative
        gradOut = gradIn * sigmoid_grad
        return gradOut
    

    def gradient2(self):
        sigmoid = self.forward(self.getPrevIn())
        return sigmoid * (1 - sigmoid)

    
    def backward2(self, grad_input):
        output = self.getPrevOut()
        return grad_input * output * (1 - output)