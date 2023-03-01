
import sys
import inspect
import random

from numpy import *
from pylab import *
from matplotlib import pyplot

import util


def visRegressedLine(X, predicted_y, w):
    ### TODO: YOUR CODE HERE
    pass
    
    
def visClassifier(X, predicted_y, w):
    ### TODO: YOUR CODE HERE
    pass
    
    
def visLoss(loss):
    ### TODO: YOUR CODE HERE
    pass
    
    
def visLikelihood(likelihood):
    ### TODO: YOUR CODE HERE
    pass


def computeClassificationAcc(org_y, predicted_y):
    '''
        Compute classification accuracy by counting how many predicted_y
        is the same to the org_y
    '''
    ### TODO: YOUR CODE HERE

    predicted_y = (predicted_y.reshape(-1) > 0.5).astype(int)
    return (org_y == predicted_y).sum() / len(org_y)


def computeAvgRegrMSError(org_y, predicted_y):
    '''
        Compute regression error by average error between predicted_y
        and org_y. Use L2 distance between two values (each eleement 
        in the vector).
    '''
    ### TODO: YOUR CODE HERE
    return ((org_y - predicted_y)**2).mean()