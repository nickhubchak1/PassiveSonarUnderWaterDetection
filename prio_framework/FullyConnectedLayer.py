import numpy as np
from .Layer import Layer


class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut):
        super().__init__()
        
        self.weights = np.random.uniform(
            low = -10**(-4),
            high = 10**(-4),
            size=(sizeIn, sizeOut)
        )
        
        self.biases = np.random.uniform(
            low = -10**(-4),
            high = 10**(-4),
            size = (1, sizeOut)
        )
    
    def getWeights(self):
        
        return self.weights
    
    def setWeights(self, weights):
        
        self.weights = weights
        
    def getBiases(self):
        return self.biases
    
    def setBiases(self, biases):
        self.biases = biases
    
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn=dataIn)       
        Y = np.dot(dataIn, self.weights) + self.biases
        self.setPrevOut(Y)
        return Y
    
    def gradient(self):
        dY = self.weights.T
        batch = np.array([dY] * self.getPrevIn().shape[0])
        return batch
        
        # batch_size = self.getPrevIn().shape[0]
        # num_inputs = self.weights.shape[0]
        # num_outputs = self.weights.shape[1]

        # # Create a tensor for the Jacobian for each sample in the batch
        # sg = np.tile(self.weights.T, (batch_size, 1, 1))  # Shape: (batch_size, num_outputs, num_inputs)
        # return sg
    
    def backward(self,gradIn):
        sg = self.gradient()
        gradOut = np.zeros((gradIn.shape[0],sg.shape[2]))
        for i in range(gradIn.shape[0]):
            gradOut[i] = np.atleast_2d(gradIn[i])@np.atleast_2d(sg[i])
            
        return gradOut
        
    
    
    def updateWeights(self, gradIn, eta):
        N = gradIn.shape[0]
        dJdb = np.sum(gradIn, axis=0) / N
        dJdW = (self.getPrevIn().T @ gradIn) / N
        self.weights -= eta*dJdW
        self.biases -= eta*dJdb
        
    def backward2(self, gradIn):
        return gradIn @ self.weights.T