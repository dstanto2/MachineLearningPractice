import torch
import calcCost


def gradientDescent(X, y, theta, alpha, iters):
    # get length of X/y
    m = y.size(0)
    for i in range(iters):
        yhat = torch.matmul(X, theta)
        ydiff = yhat - y
        ydiffx = ydiff * X[:, 1]
        # print("yhat", yhat)
        # print("ydiff", ydiff)
        # print("ydiffx", ydiffx)

        # compute new theta
        theta = torch.tensor([theta[0] - (alpha / m * torch.sum(ydiff)), theta[1] - (alpha / m * torch.sum(ydiffx))], dtype=torch.float64)
    return theta