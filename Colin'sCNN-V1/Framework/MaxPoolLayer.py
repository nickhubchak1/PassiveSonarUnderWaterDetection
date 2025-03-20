import numpy as np
from Framework.Layer import Layer

class MaxPoolLayer(Layer):
    def __init__(self, poolSize, stride):
        self.poolSize = poolSize  # Assume poolSize is a tuple (depth, height, width)
        self.stride = stride      # Assume stride is a tuple (stride_depth, stride_height, stride_width)
        self.prevIn = None        # Store input from the forward pass
        self.prevOut = None       # Store output of the forward pass
        self.indexSave = None     # Store indices of max elements for backpropagation

    def forward(self, inputTensor):

        self.prevIn = inputTensor  # Save input for backpropagation
        output, indexSave = self.maxPool(inputTensor)  # Perform 3D max-pooling
        self.prevOut = output     # Save output
        self.indexSave = indexSave  # Save max indices for backpropagation
        return output

    def maxPool(self, inputTensor):

        # Get input dimensions (batch_size, depth, height, width)
        N, D, H, W = inputTensor.shape
        poolDepth, poolHeight, poolWidth = self.poolSize
        strideDepth, strideHeight, strideWidth = self.stride

        # Calculate output dimensions
        outDepth = (D - poolDepth) // strideDepth + 1
        outHeight = (H - poolHeight) // strideHeight + 1
        outWidth = (W - poolWidth) // strideWidth + 1

        # Initialize output and indices
        output = np.zeros((N, outDepth, outHeight, outWidth))
        indexSave = np.zeros((N, outDepth, outHeight, outWidth, 3), dtype=int)

        # Perform 3D max-pooling
        for n in range(N):  # Loop over batch
            for d in range(0, D - poolDepth + 1, strideDepth):
                for i in range(0, H - poolHeight + 1, strideHeight):
                    for j in range(0, W - poolWidth + 1, strideWidth):
                        # Define the pooling region
                        region = inputTensor[
                            n, 
                            d:d+poolDepth, 
                            i:i+poolHeight, 
                            j:j+poolWidth
                        ]

                        # Get max value and index
                        maxVal = np.max(region)
                        out_d = d // strideDepth
                        out_i = i // strideHeight
                        out_j = j // strideWidth
                        output[n, out_d, out_i, out_j] = maxVal
                        max_index = np.unravel_index(np.argmax(region, axis=None), region.shape)
                        indexSave[n, out_d, out_i, out_j] = (
                            d + max_index[0],  # Map back to input tensor
                            i + max_index[1],
                            j + max_index[2]
                        )

        return output, indexSave

    def backward(self, gradOutput):

        # Initialize gradient for the input tensor
        N, D, H, W = self.prevIn.shape
        dInput = np.zeros_like(self.prevIn)

        # Get output dimensions
        outDepth, outHeight, outWidth = gradOutput.shape[1:]

        # Distribute the gradient to the max locations
        for n in range(N):  # Loop over batch
            for out_d in range(outDepth):
                for out_i in range(outHeight):
                    for out_j in range(outWidth):
                        # Retrieve the index of the max element
                        d, i, j = self.indexSave[n, out_d, out_i, out_j]
                        # Add the gradient from the output to the corresponding input
                        dInput[n, d, i, j] += gradOutput[n, out_d, out_i, out_j]

        return dInput

    def gradient(self):
        pass
