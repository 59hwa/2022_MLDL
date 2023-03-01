"""
A starting code for a linear regression model.
"""

from numpy import *

class Linear:
    """
    This class is for the linear regression model implementation.  
    """

    w = None

    def __init__(self):
        
        self.w = None
        self.eta = 1.0
        self.lam = 0.0
        self.epoch = 1000

    def setEta(self, etaVal):
        self.eta = etaVal

    def setLam(self, lamVal):
        self.lam = lamVal

    def setEpoch(self, nepoch):
        self.epoch = nepoch

    def online(self):
        ### TODO: YOU MAY MODIFY THIS
        return False

    def predict(self, X):
        """
        Perform inference
        """
        ### TODO: YOUR CODE HERE


    def train_CFS(self, X, Y):
        """
        Build a vanilla linear regressor by closed-form solution.
        """
        ### TODO: YOUR CODE HERE

    def train_ridge_CFS(self, X, Y):
        """
        Build a ridge regressor by closed-form solution.
        """
        ### TODO: YOUR CODE HERE

    def train_ridge_GD(self, X, Y):
        """
        Build a ridge regressor by gradient descent algorithm.
        """
        ### TODO: YOUR CODE HERE
