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

plt.show()




