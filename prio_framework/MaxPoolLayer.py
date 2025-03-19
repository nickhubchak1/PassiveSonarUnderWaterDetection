import numpy as np
from .Layer import Layer


class MaxPoolLayer(Layer):
    def __init__(self, pool_size, stride):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        batch_size, H, W = dataIn.shape
        
        pool_H, pool_W = self.pool_size, self.pool_size
        
        output_H = ((H - pool_H) // self.stride) + 1
        output_W = ((W - pool_W) // self.stride) + 1
        
        output = np.zeros((batch_size, output_H, output_W))
        
        self.max_indices = []
        
        for i in range(batch_size):
            for j in range(output_H):
                for k in range(output_W):
                    kernel_pos_j = j * self.stride
                    kernel_pos_k = k * self.stride
                    selected_region = dataIn[i, kernel_pos_j:kernel_pos_j + pool_H, kernel_pos_k:kernel_pos_k + pool_W]
                    output[i, j, k] = np.max(selected_region)
                    
                    max_index = np.unravel_index(np.argmax(selected_region), selected_region.shape)
                    self.max_indices.append((i, kernel_pos_j + max_index[0], kernel_pos_k + max_index[1]))
                    
        self.setPrevOut(output)
        return output
    
    def gradient(self):
        pass
    
    def backward(self, gradIn):
        dataIn = self.getPrevIn()
        batch_size, H, W = dataIn.shape
        gradOut = np.zeros_like(dataIn)
        
        index = 0
        for i in range(batch_size):
            for j in range(gradIn.shape[1]):
                for k in range(gradIn.shape[2]):
                    x_max, y_max = self.max_indices[index][1], self.max_indices[index][2]
                    gradOut[i, x_max, y_max] += gradIn[i, j, k]
                    index += 1
        
        return gradOut
