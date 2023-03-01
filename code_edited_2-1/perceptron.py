from numpy import *
import numpy as np
import time
import random

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
        self.thresh = 0

    def setEta(self, etaVal):
        self.eta = etaVal

    def setLam(self, lamVal):
        self.lam = lamVal
        
    def setMaxiter(self, niter):
        self.iter = niter
        
    def setThreshold(self, threshVal):
        self.thresh = threshVal

    def bias(self, X):
        X = np.concatenate([np.ones((len(X), 1)), X], axis=1)
        return X
    
    def sign(self,X):
        if X>self.thresh:
            return 1
        else:
            return -1
    def predict(self,X):
        y_pred = np.dot(self.bias(X),self.w)
        pre = np.reshape(y_pred,(len(self.bias(X)),1))
        for i in range(len(self.bias(X))):
            if sign(pre[i]) ==1:
                pre[i] = 1
            else: 
                pre[i]= 0
        # print(np.shape(pre))
        return pre
    
  
    def predict2(self, X):
        
        y_pred = np.dot(X,self.w[:2])
        pre = np.reshape(y_pred,(len(X),1))
        for i in range(len(X)):
            if sign(pre[i]) ==1:
                pre[i] = sign(pre[i])
            else: 
                pre[i]=0
        # print(np.shape(pre))
        return pre
    
    def train(self, X, Y):
        random.seed(time.time())
        if self.w is None:
            # self.w = np.random.rand(X.shape[1], 1)
            self.w = np.zeros((X.shape[1]+1, 1))
        w = np.zeros((len(X[0])+1, 1))
        for i in range(len(Y)):
            if Y[i] == 0:
                Y[i] = -1
        Y = Y.reshape(len(Y), 1)
        for i in range(self.iter):
            for j in range(len(self.bias(X))):
                if ((sign(np.dot(w.T, self.bias(X)[j][:, None])))*Y[j][:, None] <= 0):
                    w = w + self.eta*np.dot(self.bias(X)[j][:, None], Y[j][:, None])
        self.w = w         
        return w