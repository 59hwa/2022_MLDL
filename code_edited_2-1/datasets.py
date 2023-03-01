from numpy import *
# import util

from sklearn.datasets import load_breast_cancer
import random


class BreastCancerDataset:
    """
    X is a feature vector
    Y is the predictor variable
    """
    tr_x = None  # X (data) of training set.
    tr_y = None  # Y (label) of training set.
    val_x = None # X (data) of validation set.
    val_y = None # Y (label) of validation set.

    def __init__(self):
      
        dataset = load_breast_cancer()
        self.data_x = dataset['data']
        self.data_y = dataset['target']
        self.data_x_norm = (self.data_x-self.data_x.min())/(self.data_x.max()-self.data_x.min())

    def getDataset_reg(self):
        ## TODO: YOUR CODE HERE
        random_idx = list(range(len(self.data_x)))
        random.shuffle(random_idx)
        tr_idx = random_idx[:int(len(random_idx)*0.8)]
        val_idx = random_idx[int(len(random_idx)*0.8):]
        self.data_y = self.data_y.astype(float)
        self.tr_x = self.data_x_norm[tr_idx][:, :-1]  ### TODO: YOUR CODE HERE
        self.tr_y = self.data_x_norm[tr_idx][:, -1]  ### TODO: YOUR CODE HERE
        self.val_x = self.data_x_norm[val_idx][:, :-1] ### TODO: YOUR CODE HERE
        self.val_y = self.data_x_norm[val_idx][:, -1] ### TODO: YOUR CODE HERE

    

        return [self.tr_x, self.tr_y, self.val_x, self.val_y]
    
    
    def getDataset_cls(self):
        ### TODO: YOUR CODE HERE
        random_idx = list(range(len(self.data_x)))
        random.shuffle(random_idx)
        tr_idx = random_idx[:int(len(random_idx)*0.8)]
        val_idx = random_idx[int(len(random_idx)*0.8):]
        self.data_y = self.data_y.astype(float)
        self.tr_x = self.data_x_norm[tr_idx]  ### TODO: YOUR CODE HERE
        self.tr_y = self.data_y[tr_idx]  ### TODO: YOUR CODE HERE
        self.val_x = self.data_x_norm[val_idx] ### TODO: YOUR CODE HERE
        self.val_y = self.data_y[val_idx] ### TODO: YOUR CODE HERE

        return [self.tr_x, self.tr_y, self.val_x, self.val_y]

