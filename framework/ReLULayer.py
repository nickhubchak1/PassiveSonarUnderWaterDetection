
#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------import numpy as np
from .Layer import Layer

class ReLULayer(Layer):
    def __init__(self):
        super().__init__()
        self.dataIn = None

    def forward(self, dataIn):
        """Applies ReLU activation function."""
        self.dataIn = dataIn
        self.setPrevIn(dataIn)
        Y = np.maximum(0, dataIn)
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        """Computes element-wise gradient of ReLU."""
        if self.dataIn is None:
            raise ValueError("Cannot calculate gradient: dataIn is None")
        
        return (self.dataIn > 0).astype(float)  # Element-wise derivative