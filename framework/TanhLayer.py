#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------
import numpy as np
from .Layer import Layer

class TanhLayer(Layer):
    
    #Input: None
    #Output: None
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
         #input: dataIn, a (1 by K) matrix
        #output: A (1 by K) matrix

        self.setPrevIn(dataIn)
        Y = np.tanh(dataIn)
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        #Using batching

        tanh_output = self.getPrevOut()

        #print(tanh_output)
        batch_size, input_size = tanh_output.shape

        grad = np.zeros((batch_size, input_size, input_size))

        for i in range(batch_size):
            diag = 1 - tanh_output[i] **2
            grad[i] = np.diag(diag)
        
        return grad
    

    def gradient2(self):
        tanh_output = self.getPrevOut()
        return 1 - tanh_output ** 2

    def backward2(self, gradInput):
        return gradInput * (1 - self.getPrevOut() ** 2)