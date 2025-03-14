from Framework.Layer import Layer
import numpy as np

class LinearLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        # Linear activation simply returns the input unchanged
        return dataIn

    def gradient(self, gradOutput):
        # The gradient of a linear activation is 1, hence we return gradOutput unchanged
        return gradOutput

    def backward(self, gradOutput):
        # Implementing the backward pass for completeness
        return self.gradient(gradOutput)
