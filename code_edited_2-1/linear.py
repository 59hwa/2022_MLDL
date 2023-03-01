"""
A starting code for a linear regression model.
"""

from numpy import *
import numpy as np


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
        self.bias = True
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
        if X.shape[1] == self.w.shape[0] - 1:
            X = np.concatenate([np.ones((len(X), 1)), X], axis=1)
        return np.dot(X, self.w)
    
    def predict2(self, X):
        """
        Perform inference
        """
        ### TODO: YOUR CODE HERE
        if X.shape[1] == self.w.shape[0] - 1:
            X = np.concatenate([np.ones((len(X), 1)), X], axis=1)
        return np.dot(X, self.w[:2])

    def train_CFS(self, X, Y):
        """
        Build a vanilla linear regressor by closed-form solution.
        """
        ### TODO: YOUR CODE HERE
        if self.w is None:
            # self.w = np.random.rand(X.shape[1], 1)
            self.w = np.zeros((X.shape[1]+1, 1))
        self.w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
  
   
        
    def train_ridge_CFS(self, X, Y):
        """
        Build a ridge regressor by closed-form solution.
        """
        ### TODO: YOUR CODE HERE
        if self.w is None:
            # self.w = np.random.rand(X.shape[1], 1)
            self.w = np.zeros((X.shape[1]+1, 1))
            
        X_ict = np.c_[np.ones((X.shape[0], 1)), X]
        dimension = X_ict.shape[1]
        A = np.identity(dimension)
        A[0, 0] = 0
        biased = self.lam * A
        self.w = np.dot(np.linalg.inv(np.dot(X_ict.T,X_ict) + biased),np.dot(X_ict.T,Y))
        # self.w = self.w.reshape(-1, 1)
    
    

    def train_ridge_GD(self, X, Y):
        """
        Build a ridge regressor by gradient descent algorithm.
        """
        ### TODO: YOUR CODE HERE
        if self.w is None:
            # self.w = np.random.rand(X.shape[1]+1, 1)
            self.w = np.zeros((X.shape[1]+1, 1))
        X = np.concatenate([np.ones((len(X), 1)), X], axis=1)
        for i in range(self.epoch):
            Y_pred = self.predict(X)   
            dw = ( - ( 2 * ( X.T ).dot( Y - Y_pred ) ) +               
                ( 2 * self.lam * self.w ) )/ (X.shape[0] * X.shape[1])
            self.w = self.w - self.eta * dw.mean(axis=1).reshape(-1, 1)
        return self
