#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------
import numpy as np

class LogLoss:
    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon

    # Input: Y is an N by K matrix of target values.
    # Input: Yhat is an N by K matrix of estimated values.
    # Output: A single floating point value.
    def eval(self, Y, Yhat):
        Yhat = np.where(Yhat < self.epsilon, self.epsilon, Yhat)
        Yhat = np.where(Yhat > 1 - self.epsilon, 1 - self.epsilon, Yhat)

        loss = -np.mean(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat))
        return loss

    # Input: Y is an N by K matrix of target values.
    # Input: Yhat is an N by K matrix of estimated values.
    # Output: An N by K matrix.
    def gradient(self, Y, Yhat):
        Yhat = np.where(Yhat < self.epsilon, self.epsilon, Yhat)
        Yhat = np.where(Yhat > 1 - self.epsilon, 1 - self.epsilon, Yhat)

        grad = -(Y - Yhat) / (Yhat * (1 - Yhat))
        return grad
