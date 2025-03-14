from Framework.Layer import Layer
import numpy as np

class SoftmaxLayer(Layer):
    def __init__(self):
        self.saveOutput = None

    def forward(self, dataIn):
        exp_data = np.exp(dataIn - np.max(dataIn, axis=1, keepdims=True))
        self.saveOutput = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        return self.saveOutput

    def gradient(self):
        softmax_output = self.saveOutput
        d_input = np.empty((softmax_output.shape[0], softmax_output.shape[1], softmax_output.shape[1]))
        
        for i in range(softmax_output.shape[0]):
            s = softmax_output[i].reshape(-1, 1)
            jacobian_matrix = np.diagflat(s) - np.dot(s, s.T)
            d_input[i] = jacobian_matrix
        
        return d_input