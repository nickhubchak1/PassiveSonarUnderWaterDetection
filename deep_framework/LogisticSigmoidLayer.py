import numpy as np
from .Layer import Layer

class LogisticSigmoidLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn=dataIn)
        Y = 1 / (1 + np.exp(-1 * dataIn))
        self.setPrevOut(Y)
        return Y
    
    def gradient(self):
        eps = 10**(-7)
        Y = self.getPrevOut()
        batch = []
        dim = Y.shape[0]
        
        for i in range(dim):
            dY = np.diag(Y[i] * (1 - Y[i]) + eps)
            batch.append(dY)
        
        return np.array(batch)
    
    def backward(self,gradIn):
        sg = self.gradient()
        gradOut = np.zeros((gradIn.shape[0],sg.shape[2]))
        for i in range(gradIn.shape[0]):
            gradOut[i] = np.atleast_2d(gradIn[i])@np.atleast_2d(sg[i])
            
        return gradOut

    def gradient2(self):
        Y = self.getPrevOut()
        return Y * (1 - Y) 
    
    def backward2(self, gradIn):
        return gradIn * self.gradient2()  