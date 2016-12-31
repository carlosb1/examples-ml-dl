import pandas as pd
from pandas import Series
import numpy as np
import pylab as plt

PATH_DATASET_H5 = 'data/processed/dataset.h5'
PATH_DATASET_CSV = 'data/processed/dataset.csv'


print 'reading dataset model from...'
df =  pd.read_hdf(PATH_DATASET_H5,'table')

print 'visualizing information from dataset'


#Check most used products 
print df['product_id'].value_counts()

#Check most important client 
print df['customer_id'].value_counts()

#Check subclass
print df['product_subclass'].value_counts()

plt.rc('figure',figsize=(10,5))
fizsize_with_subplots=(10,10)
fig = plt.figure(figsize=fizsize_with_subplots)
fig.suptitle('Number of transactions for age and area')
fig_dims=(2,1)

plt.subplot2grid(fig_dims,(0,0))
df['area'].value_counts().plot(kind='bar', title='areas')

plt.subplot2grid(fig_dims,(1,0))
df['age'].value_counts().plot(kind='bar', title='ages')


fig = plt.figure(figsize=fizsize_with_subplots)
plt.suptitle('Most demanded type of products')
df['product_subclass'].value_counts().head(150).plot(kind='bar',fontsize='6', title='product subclasses')


fig = plt.figure(figsize=fizsize_with_subplots)
df.index.value_counts().head(150).plot(kind='bar',fontsize='6', title='Days with more transactions')


fig = plt.figure(figsize=fizsize_with_subplots)
amount = Series(df.groupby(df.index)['amount'].sum(),index=df.index).drop_duplicates()
amount.plot(title="Total sum of bought elements")

fig = plt.figure(figsize=fizsize_with_subplots)
bought=df['amount']*df['sales_price']
values=bought.groupby(df.index).sum()
total_benefit= Series(values,index=df.index).drop_duplicates()
total_benefit.plot(title="Total benefit")


cross = pd.crosstab(df['resident_area'],df['age_group'])
cross_pct = cross.div(cross.sum(1).astype(float), axis=0)
cross_pct.plot(kind='bar',stacked=True, title='Relation area and age group')


fig = plt.figure(figsize=fizsize_with_subplots)
cross = pd.crosstab(df['age_group'],df['product_subclass'])
cross_pct = cross.div(cross.sum(1).astype(float), axis=0)
cross_pct.plot(kind='bar',stacked=True, title='Relation product and age group')
plt.show()

fig = plt.figure(figsize=fizsize_with_subplots)
cross = pd.crosstab(df['resident_area'],df['product_subclass'])
cross_pct = cross.div(cross.sum(1).astype(float), axis=0)
cross_pct.plot(kind='bar',stacked=True, title='Relation product and age group')
plt.show()




