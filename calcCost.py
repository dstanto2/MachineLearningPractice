import torch


def calcCost(X, y, theta):
    pass
    # get length of X/y
    m = y.size(0)
    # compute yhat
    yhat = torch.matmul(X, theta)
    # compute ydiff
    ydiff = yhat - y
    # compute ydiffsquared
    ydiffsquared = torch.pow(ydiff, 2.0)
    # compute sumdiffsquared
    sumdiffsquared = torch.sum(ydiffsquared)
    #return cost
    cost = sumdiffsquared / (m * 2.0)
    return cost