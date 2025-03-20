#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------
import numpy as np
from framework import Layer

class ConvolutionalLayer(Layer):
    def __init__(self, kernel_size, num_kernels=1):
        
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd number.")

        if(num_kernels < 1):
            raise ValueError("Number of Kernels must be at least 1.")
        
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels

        #self.kernels = np.zeros((num_kernels, kernel_size, kernel_size)) #just setting to zeros 
        # Xavier Initialization for the kernels
        # Xavier Initialization with correct shape: (num_kernels, kernel_size, kernel_size)
        limit = np.sqrt(6 / (kernel_size * kernel_size + num_kernels))
        self.kernels = np.random.uniform(-limit, limit, (num_kernels, kernel_size, kernel_size))
        print("kernel_shape: ", self.kernels.shape)

    def setKernels(self, kernels):
        if self.num_kernels == 1:
            if kernels.shape == (self.kernel_size, self.kernel_size):  
                kernels = np.expand_dims(kernels, axis=0)  # Convert to (1, N, N)
        elif kernels.shape != (self.num_kernels, self.kernel_size, self.kernel_size):
            raise ValueError("Kernels shape does not match the initialized size.")
        self.kernels = kernels
        

    def getKernels(self):
        return self.kernels
    
    @staticmethod
    def crossCorrelate2D(kernel, input_matrix):

        input_h, input_w = input_matrix.shape
        kernel_h, kernel_w = kernel.shape

        if kernel_h % 2 == 0 or kernel_w % 2 == 0:
            raise ValueError("Kernel size must be odd")
        
        output_h = input_h - kernel_h + 1
        output_w = input_w - kernel_w + 1 
        output = np.zeros((output_h, output_w))

        for i in range(output_h):
            for j in range(output_w):
                region = input_matrix[i:i+kernel_h, j:j+kernel_w]
                output[i, j] = np.sum(region * kernel)
        return output
    
    def forward(self, input_tensor):
        '''
        Input tensor: incoming data (N x H x W) or (H x W)
        Output: cross-correlation output of tensor
        '''
        #print("Input tensor shape: ", input_tensor.shape)

        # Ensure input has three dimensions (N, H, W)
        if input_tensor.ndim == 2:  # If input is (H, W), reshape to (1, H, W)
            input_tensor = np.expand_dims(input_tensor, axis=0)
        N, H, W = input_tensor.shape
        kernel_h, kernel_w = self.kernels.shape[1:]
        output_h = H - kernel_h + 1
        output_w = W - kernel_w + 1
        output_tensor = np.zeros((N, self.num_kernels, output_h, output_w))
        self.setPrevIn(input_tensor)
        for n in range(N):
            for k in range(self.num_kernels):
                output_tensor[n, k] = self.crossCorrelate2D(self.kernels[k], input_tensor[n])
          # Remove num_kernels axis if only 1 kernel (reshape to (N, H_out, W_out))
        if self.num_kernels == 1:
            output_tensor = np.squeeze(output_tensor, axis=1)
        self.setPrevOut(output_tensor)
        return output_tensor

    def getKernelsDimReduce(self):
        N, M, M = self.kernels.shape
        return self.kernels.reshape(M,M)
    
    def gradient(self):
        pass
    
    def backward(self, grad_output):
        pass


    #def updateKernels(self, grad_output, learning_rate = 0.01):

        #print("gradoutput: ", grad_output)


        # # (1, M, M) K  (3,M,M)
        # # (1, H, W) X (14, H, W)
        # X = self.getPrevIn()
        # count, H, W = X.shape
        # C_out, H_grad, W_grad = grad_output.shape
        # M = self.kernel_size
        # kernel_grad = np.zeros((count, M, M))
    
        # for c in range(count):
        #     for i in range(M):
        #         for j in range(M):
        #             kernel_grad[c, i, j] = np.sum(
        #                 grad_output[count, :, :] * X[c, i:H-M+i+1, j:W-M+j+1]
        #             )
        # kernel_grad = kernel_grad - ( learning_rate * kernel_grad )


    def updateKernels(self, grad_output, learning_rate=0.01):
        """
        Update convolutional kernels using the gradient from the max pool layer.
        
        Parameters:
        - grad_output (numpy array): Gradient from the max pool layer of shape (C, H_grad, W_grad)
        - learning_rate (float): Learning rate for updating kernels, default 0.01
        """
        X = self.getPrevIn()  # Get input image matrix
        
        C, H_grad, W_grad = grad_output.shape
        #V, H, W = X.shape  # Handle (1, H, W) case implicitly
        N, M, M = self.kernels.shape if len(self.kernels.shape) == 3 else (1, *self.kernels.shape[1:])
        
        kernel_grad = np.zeros_like(self.kernels)  # Initialize kernel gradient matrix
        
        for c in range(C):  # Loop over output channels
            for i in range(M):  # Loop over kernel height
                for j in range(M):  # Loop over kernel width
                    kernel_grad[c, i, j] = np.sum(
                        grad_output[c] * X[:, i:H_grad + i, j:W_grad + j]  # Cross-correlation operation
                    )
        
        # Update the kernels using gradient descent
        self.kernels = self.kernels - ( learning_rate * kernel_grad )
        #kernel_output = self.kernels.reshape(3,3)
        #return kernel_output

    
    def specialUpdateKernels(self, grad_output, learning_rate=0.01):
        """
        Update convolutional kernels using the gradient from the max pool layer.
        
        Parameters:
        - grad_output (numpy array): Gradient from the max pool layer of shape (C, H_grad, W_grad)
        - learning_rate (float): Learning rate for updating kernels, default 0.01
        """
        X = self.getPrevIn()  # Get input image matrix
        
        C, H_grad, W_grad = grad_output.shape
        #V, H, W = X.shape  # Handle (1, H, W) case implicitly
        N, M, M = self.kernels.shape if len(self.kernels.shape) == 3 else (1, *self.kernels.shape[1:])
        
        kernel_grad = np.zeros((C, M, M))  # Initialize kernel gradient matrix and make sure we got 14 of them or C
        
        for c in range(C):  # Loop over output channels
            for i in range(M):  # Loop over kernel height
                for j in range(M):  # Loop over kernel width
                    kernel_grad[c, i, j] = np.sum(
                        grad_output[c] * X[:, i:H_grad + i, j:W_grad + j]  # Cross-correlation operation
                    )
        
        # Update the kernels using gradient descent
        self.kernels = self.kernels - ( learning_rate * np.mean(kernel_grad, axis=0, keepdims=True ))
        #kernel_output = self.kernels.reshape(3,3)
        #return kernel_output