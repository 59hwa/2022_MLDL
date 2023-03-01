"""
A starting code for a logistic regression model.
"""

from numpy import *
import numpy as np
import random


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
        self.lam = 0.01
        self.iter = 1000
        self.thresh = 0.01

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
        if X.shape[1] == self.w.shape[0] - 1:
            X = np.concatenate([np.ones((len(X), 1)), X], axis=1)
        return self.sigmoid(np.dot(X, self.w))
    
    def predict2(self, X):
        """
        Perform inference
        """
        ### TODO: YOUR CODE HERE
        if X.shape[1] == self.w.shape[0] - 1:
            X = np.concatenate([np.ones((len(X), 1)), X], axis=1)
        return self.sigmoid(np.dot(X, self.w[:2]))
    

    def train_GA(self, X, Y):
        """
        Build a logistic regression model by gradient ascent algorithm.
        """
        ### TODO: YOUR CODE HERE
        if self.w is None:
            self.w = np.zeros((X.shape[1]+1, 1))
            # self.w = np.random.rand(X.shape[1]+1, 1)
        X = np.concatenate([np.ones((len(X), 1)), X], axis=1)
        for i in range(self.iter):
            z = np.dot(X, self.w)
            y_hat = self.sigmoid(z)
            grad = np.mean((Y-y_hat.reshape(-1))*X.T, axis=1)
            self.w = self.w + self.eta * grad.reshape(-1, 1)
    
    
    def train_SGA(self, X, Y):
        """
        Build a logistic regression model by stochastic gradient ascent algorithm.
        """
        ### TODO: YOUR CODE HERE
        if self.w is None:
            self.w = np.zeros((X.shape[1]+1, 1))
            # self.w = np.random.rand(X.shape[1]+1, 1)
        X = np.concatenate([np.ones((len(X), 1)), X], axis=1)
        idxs = list(range(len(X)))
        for i in range(self.iter):
            random.shuffle(idxs)
            random_idx = idxs[:256]
            sample_X = X[random_idx]
            sample_Y = Y[random_idx]
            z = np.dot(sample_X, self.w)
            y_hat = self.sigmoid(z)
            grad = np.mean((sample_Y-y_hat.reshape(-1))*sample_X.T, axis=1)
            # print(grad)
            self.w = self.w + self.eta * grad.reshape(-1, 1)
            
            
    
    def train_reg_SGA(self, X, Y):
        """
        Build a regularized logistic regression model by stochastic gradient ascent algorithm.
        """
        ### TODO: YOUR CODE HERE
        if self.w is None:
            self.w = np.zeros((X.shape[1]+1, 1))
            # self.w = np.random.rand(X.shape[1]+1, 1)
        X = np.concatenate([np.ones((len(X), 1)), X], axis=1)
        idxs = list(range(len(X)))
        for i in range(self.iter):
            random.shuffle(idxs)
            random_idx = idxs[:256]
            sample_X = X[random_idx]
            sample_Y = Y[random_idx]
            z = np.dot(sample_X, self.w)
            y_hat = self.sigmoid(z)
            grad = np.mean((sample_Y-y_hat.reshape(-1))*sample_X.T, axis=1) 
            self.w = self.w + self.eta * (grad.reshape(-1, 1) - self.lam*np.sum(np.square(self.w)))
        
        
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    

 