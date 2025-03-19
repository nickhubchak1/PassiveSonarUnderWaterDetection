import numpy as np
from .Layer import Layer


class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn=dataIn)

        shifted = dataIn - np.max(dataIn, axis=1, keepdims=True)
        
        exponent = np.exp(shifted)
        
        Y = exponent / np.sum(exponent, axis = 1, keepdims=True)
        
        

        self.setPrevOut(Y)
        return Y
    
    def gradient(self):
        Y = self.getPrevOut()
        batch_size, num_classes = Y.shape
        
        J = np.zeros((batch_size, num_classes, num_classes))
        for i in range(batch_size):
            diag_y = np.diag(Y[i])
            outer_prod = np.outer(Y[i], Y[i])
            J[i] = diag_y - outer_prod
        
        return J
    
    
    def backward(self,gradIn):
        sg = self.gradient()
        gradOut = np.zeros((gradIn.shape[0],sg.shape[2]))
        for i in range(gradIn.shape[0]):
            gradOut[i] = np.atleast_2d(gradIn[i])@np.atleast_2d(sg[i])
            
        return gradOut