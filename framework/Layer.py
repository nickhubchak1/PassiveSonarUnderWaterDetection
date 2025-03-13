#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------
import numpy as np
import math
from scipy import signal
from abc import ABC, abstractmethod
import random

##########BASE CLASS###########
class Layer(ABC):
  def __init__(self):
    self.__prevIn = []
    self.__prevOut = []
  
  def setPrevIn(self,dataIn):
    self.__prevIn = dataIn
  
  def setPrevOut(self, out):
    self.__prevOut = out
  
  def getPrevIn(self):
    return self.__prevIn
  
  def getPrevOut(self):
    return self.__prevOut
  
  """
  def backward(self, gradIn):
    sg = self.gradient()
    gradOut = np.zeros((gradIn.shape[0],sg.shape[2]))

    for n in range(gradIn.shape[0]):
        gradOut[n] = np.atleast_2d(gradIn[n])@sg[n]
    return gradOut
  """
  def backward(self,gradIn):
    sg = self.gradient()
    
    if(sg.ndim == 3):  #tensor coming back
        gradOut = np.zeros((gradIn.shape[0],sg.shape[2]))
        for i in range(gradIn.shape[0]):
            gradOut[i] = np.atleast_2d(gradIn[i])@np.atleast_2d(sg[i])
    else:
        gradOut = np.atleast_2d(gradIn)@sg
    
    return gradOut

  @abstractmethod
  def forward(self,dataIn):
    pass
 
  @abstractmethod
  def gradient(self):
    pass

