import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import svm
from scipy.io import loadmat

spam_train = loadmat('data/spamTrain.mat')  
spam_test = loadmat('data/spamTest.mat')

X = spam_train['X']  
Xtest = spam_test['Xtest']  
y = spam_train['y'].ravel()  
ytest = spam_test['ytest'].ravel()
print spam_train
print spam_train['y']
svc = svm.SVC()  
svc.fit(X, y)  
print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))  
