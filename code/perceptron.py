"""
A starting code for a perceptron.
"""

from numpy import *
import numpy as np

class Perceptron:
    """
    This class is for the perceptron implementation 
    for binary classification problem.
    """

    def __init__(self):
        """
        Initialize our internal state.
        """
        self.w = None
        self.eta = 1.0
        self.lam = 0.0
        self.iter = 1000
        self.thresh = 0.001

    def setEta(self, etaVal):
        self.eta = etaVal

    def setLam(self, lamVal):
        self.lam = lamVal
        
    def setMaxiter(self, niter):
        self.iter = niter
        
    def setThreshold(self, threshVal):
        self.thresh = threshVal

    def predict(self, X):
        """
        Perform inference
        """
        ### TODO: YOUR CODE HERE

    def train(self, X, Y):
        """
        Build a perceptron by stochastic gradient descent algorithm.
        """
        ### TODO: YOUR CODE HERE

                
