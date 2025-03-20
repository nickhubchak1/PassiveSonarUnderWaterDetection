#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------
from framework import Layer
import numpy as np

class FlatteningLayer(Layer):

    def __init__(self):
        super().__init__()
    
    def forward(self, input_tensor):
        self.setPrevIn(input_tensor)
        N, H, W = input_tensor.shape
        self.setPrevOut(input_tensor.reshape(N, H*W, order='F')) #Using Fortran order, :()
        return self.getPrevOut()
    
    def gradient(self):
        pass

    def backward(self, grad_output):
        input_tensor = self.getPrevIn()
        N, H, W = input_tensor.shape
        grad_input = grad_output.reshape((N, H, W), order='F')
        return grad_input