import numpy as np
from .Layer import Layer

class FlatteningLayer(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        batch_size = dataIn.shape[0]
        output = dataIn.reshape(batch_size, -1, order="F")
        self.setPrevOut(output)
        return output
    
    def gradient(self):
        pass
    
    def backward(self, gradIn):
        dataIn = self.getPrevIn()
        gradOut = gradIn.reshape(dataIn.shape, order = "F")
        return gradOut
    
    