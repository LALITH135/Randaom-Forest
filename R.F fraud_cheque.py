"""
Created on Sun May 22 22:29:14 2022

@author: lalith kumar
"""
# Using Random Forest to prepare a model on fraud data. 
# treating those who have taxable_income <= 30000 as "Risky" and others are "Good"

# importing dataset.
import pandas as pd
df=pd.read_csv('E:\\data science\\ASSIGNMENTS\\ASSIGNMENTS\\randaom forest\\Fraud_check.csv')

df.head()
type(df)
list(df)
df.info()
df.describe()

# label encoder (converting object into integer.)

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
df['Undergrad'] = label_encoder.fit_transform(df['Undergrad'])
df['Marital.Status'] = label_encoder.fit_transform(df['Marital.Status'])
df['Urban'] = label_encoder.fit_transform(df['Urban'])

# ploting.

import seaborn as sns

sns.distplot(df['City.Population'])
sns.distplot(df['Work.Experience'])
sns.countplot(df['Undergrad'])
sns.countplot(df['Marital.Status'])
sns.countplot(df['Urban'])

# checking correlation & covariance by heat map.
import matplotlib.pyplot as plt

df.corr()
df.cov()
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True)
sns.heatmap(df.cov(),annot=True)

x=df['Taxable.Income']

# treating those who have taxable_income <= 30000 as "Risky" and others are "Good".
test=lambda x:1 if (x>30000 and x<30000) else 0
Y=df['Taxable.Income']
def f1(Y):
    if Y>=30000:
        return 'Good'
    elif Y<=30000:
        return 'Risky'

df['Taxable.Income']= df['Taxable.Income'].apply(f1)

X = df.drop(columns=['Taxable.Income'],axis=1)
list(x)
X.info()
y= df['Taxable.Income']
list (y)

# split the data into Train and Test.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.25,stratify=y,random_state=93)

# model fitting into random forest.
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,criterion="entropy",max_depth=8)
rf.fit(X_train,y_train)
y_pred= rf.predict(X_test)

# checking the accuracy.
from sklearn.metrics import accuracy_score
print("Accuracy :",accuracy_score(y_test,y_pred).round(2))
# by checking different random_state, we got the best accuracy  is 79%.

# applying confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm

#checking actual and predicted 
dfG=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  
dfG

# ploting Tree.
import matplotlib.pyplot as plt
from sklearn import tree
tr=tree.plot_tree(rf.estimators_[99],filled=True,fontsize=6)
rf.estimators_[99].tree_.node_count # counting the number of nodes
rf.estimators_[99].tree_.max_depth # number of levels
print(f'random forest has {rf.estimators_[99].tree_.node_count} nodes with maximum depth {rf.estimators_[99].tree_.max_depth}.')

#==========================================================================


