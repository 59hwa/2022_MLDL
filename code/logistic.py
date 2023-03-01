"""
A starting code for a logistic regression model.
"""

from numpy import *
import numpy as np

class Logistic:
    """
    This class is for the logistic regression model implementation 
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

    def train_GA(self, X, Y):
        """
        Build a logistic regression model by gradient ascent algorithm.
        """
        ### TODO: YOUR CODE HERE

        
    def train_SGA(self, X, Y):
        """
        Build a logistic regression model by stochastic gradient ascent algorithm.
        """
        ### TODO: YOUR CODE HERE


    def train_reg_SGA(self, X, Y):
        """
        Build a regularized logistic regression model by stochastic gradient ascent algorithm.
        """
        ### TODO: YOUR CODE HERE


                
