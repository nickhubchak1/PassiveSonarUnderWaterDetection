import numpy as np
from .Layer import Layer


class TanhLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn=dataIn)       
        Y = np.tanh(dataIn)
        self.setPrevOut(Y)
        return Y
    
    def gradient(self):
        Y = self.getPrevOut()
        batch = []
        dim = Y.shape[0]
        
        for i in range(dim):
            batch.append(np.diag(1 - Y[i] ** 2))
        return np.array(batch)
    
    def gradient2(self):
        Y = self.getPrevOut()
        return 1 - Y ** 2
    
    
    def backward(self,gradIn):
        sg = self.gradient()
        gradOut = np.zeros((gradIn.shape[0],sg.shape[2]))
        for i in range(gradIn.shape[0]):
            gradOut[i] = np.atleast_2d(gradIn[i])@np.atleast_2d(sg[i])
            
        return gradOut
    
    def backward2(self, gradIn):
        return gradIn * self.gradient2()