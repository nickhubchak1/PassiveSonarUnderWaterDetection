import numpy as np
from .Layer import Layer


class ReLULayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn=dataIn)
        Y = np.maximum(0, dataIn)
        self.setPrevOut(Y)
        return Y
    
    def gradient(self):
        grad = self.getPrevOut()
        dY = grad > 0
        dY = dY.astype(float)
        
        dim = grad.shape[0]
        
        return np.array([np.diag(dY[i]) for i in range(dim)])
    
    def backward2(self,gradIn):
        sg = self.gradient()
        gradOut = np.zeros((gradIn.shape[0],sg.shape[2]))
        for i in range(gradIn.shape[0]):
            gradOut[i] = np.atleast_2d(gradIn[i])@np.atleast_2d(sg[i])
            
        return gradOut
    
    