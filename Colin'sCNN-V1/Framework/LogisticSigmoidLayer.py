import numpy as np
from Framework.Layer import Layer

class LogisticSigmoidLayer:
    def __init__(self):
        self.saveOutput = None

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)  # Clipping to avoid overflow
        return 1 / (1 + np.exp(-x))

    def forward(self, dataIn):
        self.saveOutput = self.sigmoid(dataIn)

        return self.saveOutput

    def backward(self, dL_dout):
        sigmoid_output = self.saveOutput
        gradient = dL_dout * sigmoid_output * (1 - sigmoid_output)

        return gradient
