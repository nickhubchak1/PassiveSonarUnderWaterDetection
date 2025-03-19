import numpy as np
from .Layer import Layer

class LinearLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn=dataIn)
        Y = dataIn
        self.setPrevOut(Y)
        return Y
    
    def gradient(self):
        # Get the number of features (columns in the input)
        feature_dim = self.getPrevIn().shape[1]
        
        # Create a batch of identity matrices, one for each sample in the batch
        batch_size = self.getPrevIn().shape[0]
        return np.array([np.eye(feature_dim)] * batch_size)
    
    def backward(self,gradIn):
        sg = self.gradient()
        gradOut = np.zeros((gradIn.shape[0],sg.shape[2]))
        for i in range(gradIn.shape[0]):
            gradOut[i] = np.atleast_2d(gradIn[i])@np.atleast_2d(sg[i])
            
        return gradOut
    
    def backward2(self,gradIn):
        sg = self.gradient()
        gradOut = np.zeros((gradIn.shape[0],sg.shape[2]))
        for i in range(gradIn.shape[0]):
            gradOut[i] = np.atleast_2d(gradIn[i])@np.atleast_2d(sg[i])
            
        return gradOut