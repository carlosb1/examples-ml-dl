import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = os.getcwd() + '/data/ex1data1.txt'
data=pd.read_csv(path,header=None,names=['Population','Profit'])

#print data.head()
#print data.describe()

#data.plot(kind='scatter',x='Population',y='Profit', figsize=(12,8))

def computeCost(X, y, theta):
    inner = np.power(((X* theta.T) - y),2)
    return np.sum(inner) / (2,len(X))

data.insert(0,'Ones',1)
cols = data.shape[1]
X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]

X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0]))

value = computeCost(X,y,theta)
print value

def gradientDescent(X,y,theta,alpha,iters):
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X*theta.T) -y
        
        for j in range(parameters):
            term = np.multiply(error,X[:,j])
            term[0,j]= theta[0,j] - ((alpha /len(X))* np.sum(term))
        
        theta = temp
        cost[i] = computeCost(X,y,theta)

    return theta, cost
