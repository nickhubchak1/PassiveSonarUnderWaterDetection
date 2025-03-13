#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------
import numpy as np
class CrossEntropy():
    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon

        #Input: Y is an N by K matrix of target values.
    #Input: Yhat is an N by K matrix of estimated values.
    # Where N can be any integer >=1 
    #Output: A single floating point value.
    def eval(self, Y, Yhat):
        #TODO 
        Yhat_clipped = np.clip(Yhat, self.epsilon, 1 - self.epsilon)

        # Compute the cross-entropy loss
        loss = -np.sum(Y * np.log(Yhat_clipped)) / Y.shape[0]
        return loss

    #Input: Y is an N by K matrix of target values.
    #Input: Yhat is an N by K matrix of estimated values.
    #output: An N by K matrixS
    def gradient(self, Y, Yhat):
        Yhat_clipped = np.clip(Yhat, self.epsilon, 1 - self.epsilon)

        # Compute the gradient
        grad = -Y / Yhat_clipped
        return grad