#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------
import numpy as np
from .Layer import Layer

class InputLayer(Layer):
    def __init__(self, input_shape):
        super().__init__()
        self.stdX = None
        self.meanX = None
        self.input_shape = input_shape

    def forward(self, dataIn):
        if self.meanX is None or self.stdX is None:
            self.meanX = np.mean(dataIn, axis=0)
            self.stdX = np.std(dataIn, axis=0, ddof=1)
            self.stdX[self.stdX == 0] = 1  # Prevent division by zero
        
        zscore = (dataIn - self.meanX) / self.stdX
        return zscore

    def gradient(self):
        pass  # No gradients needed for input layer