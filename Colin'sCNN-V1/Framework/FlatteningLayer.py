import numpy as np
from Framework.Layer import Layer

class FlatteningLayer(Layer):
    def __init__(self):
        super().__init__()
        self.prevIn = None
        self.prevOut = None

    def forward(self, inputTensor):
        self.prevIn = inputTensor
        # Flatten the tensor into a matrix using column-major order and ensure integer conversion
        result = inputTensor.reshape(inputTensor.shape[0], -1, order='F')
        self.prevOut = result
        return result

    def backward(self, gradOutput):
        # Reshape the gradient output to match the original input tensor shape
        return gradOutput.reshape(self.prevIn.shape, order='F')

    def gradient(self):
        pass
