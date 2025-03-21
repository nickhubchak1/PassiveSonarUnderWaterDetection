import numpy as np
from .Layer import Layer


class ConvolutionalLayer(Layer):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernels = np.random.randn(kernel_size, kernel_size).astype(np.float64) * 0.01
        
    def setKernels(self, kernel):
        self.kernels = kernel.astype(np.float64)
        
    def getKernels(self):
        return self.kernels
    
    @staticmethod
    def crossCorrelate2D(kernel, matrix):
        H, W = matrix.shape
        kH, kW = kernel.shape
        
        output = np.zeros((H - kH + 1, W - kW + 1), dtype=np.float64)
        
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = np.sum(matrix[i:i+kH, j:j+kW] * kernel)
        
        return output
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        batch_size, H, W = dataIn.shape
        kH, kW = self.kernels.shape
        output = np.zeros((batch_size, H - kH + 1, W - kW + 1), dtype=np.float64)
        
        for i in range(batch_size):
            output[i] = ConvolutionalLayer.crossCorrelate2D(matrix = dataIn[i], kernel = self.kernels)
            
        self.setPrevOut(output)
        
        return output
    
    def gradient(self, gradIn, dataIn):
        batch_size, H, W = gradIn.shape
        kH, kW = self.kernels.shape
        gradOut = np.zeros_like(self.kernels, dtype=np.float64)
        for i in range(batch_size):
            for j in range(H):
                for k in range(W):
                    gradOut += gradIn[i, j, k] * dataIn[i, j:j+kH, k:k+kW]
                    
        return gradOut
    
    def backward(self, gradIn):
        dataIn = self.getPrevIn()
        batch_size, H, W = gradIn.shape
        gradOut = self.gradient(gradIn, dataIn)
        # self.kernels -= learning_rate * (kernel_grad / batch_size).astype(np.float64)
        return gradOut
    
    def updateKernels(self, gradIn, learning_rate):
        batch_size, H, W = gradIn.shape
        kernel_grad =  self.backward(gradIn)
        
        self.kernels -= learning_rate * (kernel_grad / batch_size).astype(np.float64)
        

        
        
    


        
        