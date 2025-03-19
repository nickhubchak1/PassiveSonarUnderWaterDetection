import numpy as np
from .Layer import Layer

class LogLoss():
    def __init__(self):
        super().__init__()
        
    def eval(self, Y, Yhat):
        eps = 1e-7
        Y =  -np.mean(Y * np.log(Yhat + eps) + (1 - Y) * np.log(1 - Yhat + eps))
        return Y
    
    def gradient(self, Y, Yhat):
        eps = 1e-7
        
        dY = ((1 - Y) / (1 - Yhat + eps)) - (Y / (Yhat + eps))
        return dY
    
    def gradient2(self, Y, Yhat):
        eps = 1e-7
        
        dY = ((1 - Y) / (1 - Yhat + eps)) - (Y / (Yhat + eps))
        return dY