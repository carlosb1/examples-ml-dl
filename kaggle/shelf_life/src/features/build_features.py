import pandas as pd
from pandas import Series
import numpy as np
import pylab as plt

PATH_DATASET_H5 = 'data/processed/dataset.h5'
PATH_FEATURES_ASSET_H5 = 'data/interim/asset.h5'
PATH_FEATURES_BENEFITS_H5 = 'data/interim/benefit.h5'

print 'reading dataset model from...'
df =  pd.read_hdf(PATH_DATASET_H5,'table')
df_benefit=df.drop(['customer_id','age_group','resident_area','product_subclass','asset','sales_price','amount'],axis=1)
df_benefit['benefit']=df['sales_price']*df['amount']
df_asset=df.drop(['customer_id','age_group','resident_area','product_subclass'],axis=1)

print 'clean missing data'
df_benefit.fillna(0)
df_asset.fillna(0)

print "save dataset in h5 format..."
dataframe_benefit_df = pd.DataFrame(df_benefit)
dataframe_benefit_df.to_hdf(PATH_FEATURES_BENEFITS_H5,'table',mode='w')

dataframe_asset_df = pd.DataFrame(df_asset)
dataframe_asset_df.to_hdf(PATH_FEATURES_ASSET_H5,'table',mode='w')

plt.rc('figure',figsize=(10,5))                                
fizsize_with_subplots=(10,10)

fig = plt.figure(figsize=fizsize_with_subplots)
plt.scatter(df_benefit['area'], df_benefit['benefit']
plt.show()

fig = plt.figure(figsize=fizsize_with_subplots)
plt.scatter(df_benefit['resident_area'], df_benefit['benefit']
plt.show()

fig = plt.figure(figsize=fizsize_with_subplots)
plt.scatter(df_asset['area'], df_asset['asset'])
plt.show()

fig = plt.figure(figsize=fizsize_with_subplots)
plt.scatter(df_asset['age'], df_asset['asset'])
plt.show()




