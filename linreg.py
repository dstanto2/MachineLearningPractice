

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import calcCost as c
import gradientDescent as gd


##### load data #####
current_directory = os.getcwd()
data = pd.read_csv(os.getcwd() + '/ex1data1.txt')
data.columns = ['X', 'y']
# print(data)
X = torch.tensor([np.array(torch.ones(data.shape[0])), np.array(torch.tensor(data['X'].values))]).transpose(0, 1)
y = torch.tensor(data['y'].values)

##### set parameters #####
# theta = torch.squeeze(torch.tensor(torch.zeros(2, 1), dtype=torch.float64))
theta = torch.tensor([0, 0], dtype=torch.float64)

# print(theta.size())

# print(X.dtype)
# print(y.dtype)
# print(theta.dtype)



##### compute cost #####
cost = c.calcCost(X, y, theta)
print("cost", cost)

##### gradient descent #####
iters = 1500
alpha = 0.01
optimizedTheta = gd.gradientDescent(X, y, theta, alpha, iters)
print("optimized theta", optimizedTheta)
print("optimized cost", c.calcCost(X, y, optimizedTheta))

##### compute optimized cost #####

##### visualize data and predictions #####




