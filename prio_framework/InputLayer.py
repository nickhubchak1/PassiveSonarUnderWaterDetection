from .Layer import Layer
import numpy as np

class InputLayer(Layer):
    def __init__(self, dataIn): # dataIn is an NxD matrix
        super().__init__()
        
        # print(dataIn.shape)
        
        self.meanX = np.mean(dataIn, axis = 0)
        self.stdX = np.std(dataIn, axis = 0, ddof = 1)
        # self.stdX = np.where(self.stdX != 0, self.stdX, 1)        
        # if self.stdX == 0:
        #     self.stdX = 1
        
        self.stdX[self.stdX == 0] = 1
        
    def forward(self, dataIn):
        self.setPrevIn(dataIn=dataIn)
        # Z-score the data:
        Y = (dataIn - self.meanX) / self.stdX       
        
        self.setPrevOut(Y)
        return Y
    
    def gradient(self):
        pass
        