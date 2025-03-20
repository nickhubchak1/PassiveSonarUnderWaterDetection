#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------
import numpy as np
from .Layer import Layer
class AddLayer(Layer):
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2
    
    def forward(self, X1, X2):
        return X1 + X2
    
    def backward2(self, grad):
        return grad, grad
    
    def gradient():
        pass