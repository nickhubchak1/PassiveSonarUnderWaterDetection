#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------
import numpy as np
from .Layer import Layer

class LinearLayer(Layer):
    #Input: None
    #Output: None

    def __init__(self):
        super().__init__()
        self.dataIn = None #Going to store the variable in the class

    def forward(self, dataIn):
        #TODO
         #input: dataIn, a (1 by K) matrix
        #output: A (1 by K) matrix
        self.dataIn = dataIn
        return dataIn

    def gradient(self, gradient):
        
        if self.dataIn is None:
            raise ValueError("Cannot calc gradient for Linear: the dataIn is still set to None")
        return gradient
    
    def backward2(self, gradOutput):
        # Implementing the backward pass for completeness
        return self.gradient(gradOutput)