import numpy as np
from .Layer import Layer

class CrossEntropy():
    def __init__(self):
        super().__init__()
        
    def eval(self, Y, Yhat):
        eps = 10 ** (-7)
        Y = -np.mean(np.sum(Y * np.log(Yhat + eps), axis = 1))
        return Y
    
    def gradient(self, Y, Yhat):
        eps = 10 ** (-7)
        dY = - Y / (Yhat + eps)
        return dY