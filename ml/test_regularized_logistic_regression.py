import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os
path = os.getcwd() + '/data/ex2data2.txt'
data=pd.read_csv(path,header=None,names=['Test 1','Test 2','Accepted'])
positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]
fix, ax = plt.subplots(figsize=(12,8))

ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')

ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_xlabel('Test 2 Score')

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

#visualize sigmoid function
nums = np.arange(-10,10,step=1)
fig , ax = plt.subplots(figsize=(12,8))
ax.plot(nums,sigmoid(nums),'r')

def cost(theta,X,y):
	theta = np.matrix()
	x = np.matrix(X)
	y = np.matrix(y)
	first = np.multiply(-y, np.log(sigmoid(X*theta.T)))
	second = np.multiply((1-y), np.log(1-sigmoid(X*theta.T)))
	
	return np.sum(first - second) / (len(X))

data.insert(0,'Ones',1)

cols = data.shape[1]

plt.show()

	

	
