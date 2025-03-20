#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------
from framework import Layer
import numpy as np

class MaxPoolLayer(Layer):

    def __init__(self, pool_size, stride):
        '''
        assume width and height are the same
        '''
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_tensor):

        #print("Input tensor shape: ", input_tensor.shape)

        # Ensure input has three dimensions (N, H, W)
        if input_tensor.ndim == 2:  # If input is (H, W), reshape to (1, H, W)
            input_tensor = np.expand_dims(input_tensor, axis=0)

        N, H, W = input_tensor.shape

        # Ensure pooling is valid
        if H < self.pool_size or W < self.pool_size:
            raise ValueError(f"Pooling window size {self.pool_size}x{self.pool_size} is too large for input size {H}x{W}.")

        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        # Prevent empty output issue
        if out_h <= 0 or out_w <= 0:
            print("Warning: Pooling output is empty. Adjust pool size or stride.")
            return np.zeros((N, 1, 1))  # Return at least one output value

        output_tensor = np.zeros((N, out_h, out_w))
        self.max_indices = np.zeros((N, out_h, out_w, 2), dtype=int)
        self.setPrevIn(input_tensor)

        for n in range(N):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    w_start = j * self.stride
                    region = input_tensor[n, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size]

                    max_val = np.max(region)
                    max_idx = np.unravel_index(np.argmax(region), region.shape)

                    output_tensor[n, i, j] = max_val
                    self.max_indices[n, i, j] = (h_start + max_idx[0], w_start + max_idx[1])

        self.setPrevOut(output_tensor)
        return output_tensor
    
    def gradient(self):
        pass

    def backward(self, grad_output):
        
        input_tensor = self.getPrevIn()
        N, H, W = input_tensor.shape
        grad_input = np.zeros_like(input_tensor)

        out_h, out_w = grad_output.shape[1], grad_output.shape[2]

        for n in range(N):
            for i in range(out_h):
                for j in range(out_w):
                    h_idx, w_idx = self.max_indices[n, i, j]

                    grad_input[n, h_idx, w_idx] += grad_output[n, i, j]

        return grad_input

    