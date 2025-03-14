import numpy as np
from Framework.Layer import Layer

class InputLayer(Layer):
    def __init__(self, dataIn):
        self.meanX = np.mean(dataIn, axis=0)
        self.stdX = np.std(dataIn, axis=0, ddof=1)  # Use N-1, suggested in discord by professor.
        self.stdX[self.stdX == 0] = 1

    def forward(self, dataIn):
        """
        Computes the z-scored version of the input data.
        
        Parameters:
        dataIn (numpy.ndarray): A (1 by D) matrix representing the input data.
        
        Returns:
        numpy.ndarray: A (1 by D) matrix of the z-scored input data.
        """
        
        zscored_data = (dataIn - self.meanX) / self.stdX
        saveInput = dataIn
        saveOutput = zscored_data
        return zscored_data

    def gradient(self):
        pass  # Come back to this later on.
