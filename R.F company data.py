"""
Created on Sun May 22 19:20:38 2022

@author: lalith kumar
"""
# importing the datasets. 

import pandas as pd
df=pd.read_csv('E:\\data science\\ASSIGNMENTS\\ASSIGNMENTS\\randaom forest\\Company_Data.csv')

df.shape
df.head()
list(df)
type(df)
df.info()
df.describe()

# label encoder (for categorical data)

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
df['ShelveLoc'] = label_encoder.fit_transform(df['ShelveLoc'])
df['Urban'] = label_encoder.fit_transform(df['Urban'])
df['US'] = label_encoder.fit_transform(df['US'])

#Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’)
# split x&y variables
X = df.iloc[:,1:11] 
Y = df['Sales']

X.head()
Y.head()
list(X)
list(Y)

# taking the sales average for our convinence,to convert into categorical variable.  
df['Sales'].mean() #  7.496325

x=df['Sales']

# converting the sales data into categrocial as 'low' and 'high'

test=lambda x:1 if (x>7.49 and x<7.49) else 0

def f1 (x) :
    if x<=7.49:
        return' low'
    elif x>=7.49:
        return 'high'

df['new sales']= df['Sales'].apply(f1)

y=df['new sales']
list(y)

# again spliting x&y (new sales data after converting into categorical )
X = df.iloc[:,1:11]  #Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’)
Y = df['new sales']

# split the data into Train and Test(test=0.25).
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.25,stratify=y,random_state=93)

# model fitting in random forest.
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,criterion="entropy",max_depth=8)
rf.fit(X_train,y_train)
y_pred= rf.predict(X_test)

# checking the accuracy score.
from sklearn.metrics import accuracy_score
print("Accuracy :",accuracy_score(y_test,y_pred).round(2))

# by checking different random_state, we got the best accuracy between 79 to84%.

# applying confusion matrix.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm

#checking actual and predicted. 
dfG=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  
dfG

# ploting tree.
from sklearn import tree
tr=tree.plot_tree(rf.estimators_[99],filled=True,fontsize=6)
rf.estimators_[99].tree_.node_count # counting the number of nodes
rf.estimators_[99].tree_.max_depth # number of levels
print(f'random forest has {rf.estimators_[99].tree_.node_count} nodes with maximum depth {rf.estimators_[99].tree_.max_depth}.')

#=============================================================================

