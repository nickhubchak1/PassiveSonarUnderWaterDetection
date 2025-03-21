import numpy as np
from .Layer import Layer

class SquaredError():
    def __init__(self):
        super().__init__()
        
    def eval(self, Y, Yhat):
        return np.mean((Y - Yhat)**2)
    
    def gradient(self, Y, Yhat):
        grad = -2 * (Y.reshape(-1, 1) - Yhat)
        return grad