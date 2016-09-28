import pandas as pd
import numpy as np
import pylab as plt


########### UNDERSTANDING DATA ############


#set up plots to visualize info
plt.rc('figure', figsize=(10, 5))
fizsize_with_subplots = (10,10)
bin_size=10

df_train = pd.read_csv('../data/titanic/train.csv')

#previsualize values
print "HEAD values: "
print df_train.head()
print "--------------------------------------"
print "TAIL values: "
print df_train.tail()
print "--------------------------------------"
print df_train.dtypes
print df_train.info()
print df_train.describe()


fig = plt.figure(figsize=fizsize_with_subplots)
fig_dims = (3,2)

plt.subplot2grid(fig_dims,(0,0))
df_train['Survived'].value_counts().plot(kind='bar',title='Death and Survival Counts')


plt.subplot2grid(fig_dims,(0,1))
df_train['Pclass'].value_counts().plot(kind='bar',title='Passenger Class Counts')

plt.subplot2grid(fig_dims,(1,0))
df_train['Sex'].value_counts().plot(kind='bar', title='Gender Counts')

plt.xticks(rotation=0)
plt.subplot2grid(fig_dims,(1,1))
df_train['Embarked'].value_counts().plot(kind='bar',title='Ports of Embarkation Counts')

plt.subplot2grid(fig_dims,(2,0))
df_train['Age'].hist()
plt.title('Age Histogram')



pclass_xt = pd.crosstab(df_train['Pclass'],df_train['Survived'])
print pclass_xt


#Normalize the cross tab to sum to 1:

#Feature Passenger classes
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(float),axis=0)
pclass_xt_pct.plot(kind='bar',
                stacked=True,
                title='Survival Rate by Passenger Classes')

plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')

#Feature Sex
sexes=sorted(df_train['Sex'].unique())
genders_mapping = dict(zip(sexes,range(0,len(sexes)+1)))
print genders_mapping

df_train['Sex_Val']= df_train['Sex'].map(genders_mapping).astype(int)
print df_train.head()

#Get the unique values of Pclass:
passenger_classes = sorted(df_train['Pclass'].unique())
for p_class in passenger_classes:
    print 'M: ', p_class, len(df_train[(df_train['Sex']=='male') &
                    (df_train['Pclass']== p_class)])
    print 'F: ',p_class, len(df_train[(df_train['Sex'] == 'female') & 
                    (df_train['Pclass'] ==p_class)])


#Info survival rate for gender
females_df = df_train[df_train['Sex'] == 'female']

females_xt = pd.crosstab(females_df['Pclass'],df_train['Survived'])
females_xt_pct = females_xt.div(females_xt.sum(1).astype(float),axis=0)
females_xt_pct.plot(kind='bar',stacked=True,title='Female Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')



males_df = df_train[df_train['Sex'] == 'male']
males_xt = pd.crosstab(males_df['Pclass'], df_train['Survived'])
males_xt_pct = males_xt.div(males_xt.sum(1).astype(float), axis=0)
males_xt_pct.plot(kind='bar', stacked=True, title='Male Survival Rate by Passegner Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')

df_train[df_train['Embarked'].isnull()]

#Info embarked people
embarked_locs = sorted(df_train['Embarked'].unique())
embarked_locs_mapping = dict(zip(embarked_locs,range(0,len(embarked_locs)+1)))
print embarked_locs_mapping


df_train['Embarked_Val'] = df_train['Embarked'].map(embarked_locs_mapping).astype(int)
print df_train.head()

df_train['Embarked_Val'].hist(bins=len(embarked_locs), range=(0,3))
plt.title('Port of Embarkation Histogram')
plt.xlabel('Port of Embarkation')
plt.ylabel('Count')


if len(df_train[df_train['Embarked'].isnull()] > 0):
    df_train.replace({'Embarked_Val': { embarked_locs_mapping[np.nan]: embarked_locs_mapping['S']}
                    },
                    inplace=True)

embarked_locs = sorted(df_train['Embarked_Val'].unique())
print embarked_locs

embarked_val_xt = pd.crosstab(df_train['Embarked_Val'], df_train['Survived'])
embarked_val_xt_pct = embarked_val_xt.div(embarked_val_xt.sum(1).astype(float),axis=0)
embarked_val_xt_pct.plot(kind='bar',stacked=True)
plt.title('Survival Rate by Port of Embarkation')
plt.xlabel('Port of Embarkation')
plt.ylabel('Survival Rate')



fig = plt.figure(figsize=fizsize_with_subplots)

rows=2
cols=3
col_names=('Sex_Val','Pclass')

for portIdx in embarked_locs:
    for colIdx in range(0,len(col_names)):
        plt.subplot2grid((rows,cols),(colIdx,portIdx-1))
        df_train[df_train['Embarked_Val']==portIdx][col_names[colIdx]].value_counts().plot(kind='bar')

df_train=pd.concat([df_train,pd.get_dummies(df_train['Embarked_Val'],prefix='Embarked_Val')],axis=1)
print df_train

#Age 

df_train[df_train['Age'].isnull()][['Sex','Pclass','Age']].head()
df_train['AgeFill']= df_train['Age']
df_train['AgeFill'] = df_train['AgeFill'].groupby([df_train['Sex_Val'],df_train['Pclass']]).apply(lambda x: x.fillna(x.median()))

print len(df_train[df_train['AgeFill'].isnull()])

fig,axes = plt.subplots(2,1,figsize=fizsize_with_subplots)
df1=df_train[df_train['Survived']==0]['Age']
df2=df_train[df_train['Survived']==1]['Age']

max_age = max(df_train['AgeFill'])
axes[0].hist([df1, df2], 
                     bins=max_age / bin_size, 
                                  range=(1, max_age), 
                                               stacked=True)
axes[0].legend(('Died', 'Survived'), loc='best')
axes[0].set_title('Survivors by Age Groups Histogram')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')
#
## Scatter plot Survived and AgeFill
#axes[1].scatter(df_train['Survived'], df_train['AgeFill'])
#axes[1].set_title('Survivors by Age Plot')
#axes[1].set_xlabel('Survived')
#axes[1].set_ylabel('Age')


for pclass in passenger_classes:
    df_train.AgeFill[df_train.Pclass == pclass].plot(kind='kde')

plt.title('Age Density Plot by Passenger Class')
plt.xlabel('Age')
plt.legend(('1st Class', '2nd Class', '3rd Class'), loc='best')


# Set up a grid of plots
fig = plt.figure(figsize=fizsize_with_subplots) 
fig_dims = (3, 1)

# Plot the AgeFill histogram for Survivors
plt.subplot2grid(fig_dims, (0, 0))
survived_df = df_train[df_train['Survived'] == 1]
survived_df['AgeFill'].hist(bins=max_age / bin_size, range=(1, max_age))

# Plot the AgeFill histogram for Females
plt.subplot2grid(fig_dims, (1, 0))
females_df = df_train[(df_train['Sex_Val'] == 0) & (df_train['Survived'] == 1)]
females_df['AgeFill'].hist(bins=max_age / bin_size, range=(1, max_age))

# Plot the AgeFill histogram for first class passengers
plt.subplot2grid(fig_dims, (2, 0))
class1_df = df_train[(df_train['Pclass'] == 1) & (df_train['Survived'] == 1)]
class1_df['AgeFill'].hist(bins=max_age / bin_size, range=(1, max_age))


#Family size
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']
print df_train.head()

fig = plt.figure(figsize=fizsize_with_subplots) 

df_train['FamilySize'].hist()
plt.title('Family Size Histogram')

fig = plt.figure(figsize=fizsize_with_subplots) 
# Get the unique values of Embarked and its maximum
family_sizes = sorted(df_train['FamilySize'].unique())
family_size_max = max(family_sizes)

df1 = df_train[df_train['Survived'] == 0]['FamilySize']
df2 = df_train[df_train['Survived'] == 1]['FamilySize']
plt.hist([df1, df2], 
                 bins=family_size_max + 1, 
                range=(0, family_size_max), 
                stacked=True)

plt.legend(('Died', 'Survived'), loc='best')
plt.title('Survivors by Family Size')


plt.show()


########### FINAL DATA PREPARATION FOR MACHINE LEARNING ############

#Move dataframe to array

df_train.dtypes[df_train.dtypes.map(lambda x: x == 'object')]
df_train = df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'],axis=1)
df_train = df_train.drop(['Age','SibSp','Parch','PassengerId','Embarked_Val'],axis=1)
print df_train.dtypes


train_data = df_train.values
print train_data

def clean_data(df, drop_passenger_id):
    
    sexes=sorted(df['Sex'].unique())
    
    genders_mapping=dict(zip(sexes,range(0,len(sexes)+1)))

    # Transform Sex from a string to a number representation
    df['Sex_Val'] = df['Sex'].map(genders_mapping).astype(int)
    # Get the unique values of Embarked
    embarked_locs = sorted(df['Embarked'].unique())
    # Generate a mapping of Embarked from a string to a number representation        
    embarked_locs_mapping = dict(zip(embarked_locs, 
    range(0, len(embarked_locs) + 1)))
    # Transform Embarked from a string to dummy variables
    df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked_Val')], axis=1)
    # Fill in missing values of Embarked
    # Since the vast majority of passengers embarked in 'S': 3, 
    # we assign the missing values in Embarked to 'S':
    if len(df[df['Embarked'].isnull()] > 0):
        df.replace({'Embarked_Val' :
            { embarked_locs_mapping[np.nan] : embarked_locs_mapping['S']
                }
            },
            inplace=True)
    # Fill in missing values of Fare with the average Fare
    if len(df[df['Fare'].isnull()] > 0):
        avg_fare = df['Fare'].mean()
        df.replace({ None: avg_fare }, inplace=True)
    
    # To keep Age in tact, make a copy of it called AgeFill 
    # that we will use to fill in the missing ages:
    df['AgeFill'] = df['Age']

    df['AgeFill'] = df['AgeFill'].groupby([df['Sex_Val'],df['Pclass']]).apply(lambda x: x.fillna(x.median()))
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df = df.drop(['Name','Sex','Ticket','Cabin','Embarked'],axis=1)

    df = df.drop(['Age','SibSp','Parch'],axis=1)

    if drop_passenger_id:
        df = df.drop(['PassengerId'],axis=1)
    return df


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)

train_features=train_data[:,1:]

train_target=train_data[:,0]

clf = clf.fit(train_features,train_target)

score = clf.score(train_features,train_target)
print "Mean accuracy of Random Forest: {0}".format(score)


df_test=pd.read_csv('../data/titanic/test.csv')
print df_test.head()

df_test = clean_data(df_test,drop_passenger_id=False)
test_data = df_test.values

#Get the test data features, skipping the first column 'PassengerId'
test_x=test_data[:,1:]

#Predict the survirval values for the test data
test_y=clf.predict(test_x)

df_test['Survived'] = test_y
df_test[['PassengerId','Survived']].to_csv('../data/titanic/results-rf.csv',index=False)


#Evaluate Model accuracy

from sklearn import metrics
from sklearn.cross_validation import train_test_split

train_x, test_x, train_y, test_y = train_test_split(train_features,train_target,test_size=0.20, random_state=0)
print (train_features.shape, train_target.shape)
print (train_x.shape, train_y.shape)
print (test_x.shape, test_y.shape)


#Accuracy
clf = clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)
from sklearn.metrics import accuracy_score
print("Accuracy: %.2f" %(accuracy_score(test_y,predict_y)))


model_score = clf.score(test_x,test_y)
print ("Model Score %.2f \n"  % (model_score))

confusion_matrix  = metrics.confusion_matrix(test_y,predict_y)
print ("Confusion Matrix ",confusion_matrix)

print("       Predicted")
print("       |  0  |  1  |")
print("       |-----|-----|")
print("    0  | %3d | %3d |" % (confusion_matrix[0,0], confusion_matrix[0,1]))
print("Actual |-----|-----|")
print("    1  | %3d | %3d |") % (confusion_matrix[1,0],confusion_matrix[1,1])
print("       |-----|-----|")


from sklearn.metrics import classification_report
print (classification_report(test_y,predict_y,target_names=['Not survived','Survived']))
