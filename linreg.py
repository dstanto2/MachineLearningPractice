import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt




##### load data #####
data = pd.read_csv('/home/dstanto2/Documents/GettingAJob/Online Courses/Linear Regression Assignment/ex1data1.txt')
data.columns = ['X', 'y']

X = torch.cat(torch.ones(data.shape[0]), torch.tensor(data['X'].values))
y = torch.tensor(data['y'].values)

##### set parameters #####
theta = torch.tensor([0, 0])
iterations = 1500
alpha = 0.01
print(theta)

##### compute cost #####

##### gradient descent #####

##### compute optimized cost #####

##### visualize data and predictions #####




