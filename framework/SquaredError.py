#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------
import numpy as np
class SquaredError():
    
    #Input: Y is an N by K matrix of target values.
    #Input: Yhat is an N by K matrix of estimated values.
    # Where N can be any integer >=1 
    #Output: A single floating point value.
    def eval(self, Y, Yhat):
       return np.mean((Y - Yhat)**2)
       
    #Input: Y is an N by K matrix of target values.
    #Input: Yhat is an N by K matrix of estimated values.
    #output: An N by K matrixS
    def gradient(self, Y, Yhat):
         return np.atleast_2d(-2*(Y - Yhat) )

