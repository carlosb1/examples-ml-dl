import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  
from scipy.io import loadmat

def pca(X):
	X= (X-X.mean()) / X.std()
	X = np.matrix(X)
	cov = (X.T * X) / X.shape[0]

	U, S, V = np.linalg.svd(cov)

	return U, S, V

def project_data(X, U, k):
	U_reduced =U[:,:k]
	return np.dot(X,U_reduced)


def recover_data(Z,U,k):
	U_reduced = U[:,:k]
	return np.dot(Z,U_reduced.T)





data = loadmat('data/ex7data1.mat')
X=data['X']

fig,ax  = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1])

U, S, V = pca(X)
print U,S,V
Z = project_data(X,U,1)
print Z
X_recovered = recover_data(Z,U,1)
print X_recovered




fig,ax  = plt.subplots(figsize=(12,8))
ax.scatter(X_recovered[:,0], X_recovered[:,1])

plt.show()




