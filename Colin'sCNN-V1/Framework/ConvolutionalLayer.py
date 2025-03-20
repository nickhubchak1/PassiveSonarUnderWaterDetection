import numpy as np
from Framework.Layer import Layer
from numba import jit

class ConvolutionalLayer(Layer):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size  # Assuming kernel_size is a tuple (depth, height, width)
        self.kernel = None  # Initialize kernel with provided array
        self.prevIn = None  # Store the forward pass input for backprop
        self.prevOut = None  # Store the forward pass output for backprop

    def setKernels(self, mtxIn):
        # Ensure the kernel is a 3D NumPy array
        
        self.kernel = np.array(mtxIn, dtype=float)  # Ensure float dtype
        self.kernel_size = self.kernel.shape  # Automatically infer the kernel shape
        
    def getKernels(self):
        return self.kernel

    @staticmethod
    @jit(nopython=True, parallel=True)
    def crossCorrelate3D(inputTensor, kernel):
        # Input and kernel dimensions
        input_shape = inputTensor.shape
        kernel_shape = kernel.shape

        # Calculate output dimensions
        output_shape = (
            input_shape[0] - kernel_shape[0] + 1,
            input_shape[1] - kernel_shape[1] + 1,
            input_shape[2] - kernel_shape[2] + 1
        )
        
        # Initialize the output array as complex type
        output = np.zeros(output_shape, dtype=float)

        # Perform 3D cross-correlation
        for d in range(output_shape[0]):
            for i in range(output_shape[1]):
                for j in range(output_shape[2]):
                    region = inputTensor[
                        d:d + kernel_shape[0],
                        i:i + kernel_shape[1],
                        j:j + kernel_shape[2]
                    ]
                    # Explicitly handle real and imaginary parts
                    output[d, i, j] = np.sum(region * kernel)
        
        return output


    def forward(self, inputTensor):
        self.prevIn = inputTensor

        # Determine output dimensions
        batch_size = inputTensor.shape[0]
        depth = inputTensor.shape[1] - self.kernel_size[0] + 1
        height = inputTensor.shape[2] - self.kernel_size[1] + 1
        width = inputTensor.shape[3] - self.kernel_size[2] + 1

        # Initialize output with complex dtype
        output = np.zeros((batch_size, depth, height, width), dtype=complex)

        # Apply 3D convolution for each input in the batch
        for n in range(batch_size):
            output[n] = self.crossCorrelate3D(inputTensor[n], self.kernel)

        self.prevOut = output
        return output

    def gradient(self):
        pass

    def updateKernels(self, backGrad, lr):
        kernel_grad = np.zeros_like(self.kernel, dtype=float)
        batch_size = self.prevIn.shape[0]  # Assuming batch dimension is first

        for b in range(batch_size):
            for d in range(backGrad.shape[1]):
                for i in range(backGrad.shape[2]):
                    for j in range(backGrad.shape[3]):
                        # Extract the input patch
                        d_start, d_end = d, d + self.kernel_size[0]
                        i_start, i_end = i, i + self.kernel_size[1]
                        j_start, j_end = j, j + self.kernel_size[2]

                        input_patch = self.prevIn[b, d_start:d_end, i_start:i_end, j_start:j_end]

                        # Accumulate kernel gradient
                        kernel_grad += input_patch * backGrad[b, d, i, j]

        # Update the kernel using the gradient and learning rate
        self.kernel -= lr * kernel_grad / batch_size  # Normalize by batch size
