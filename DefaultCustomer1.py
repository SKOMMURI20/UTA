#!/usr/bin/env python
# coding: utf-8

# # Data Analyis on Customers defaulted on loans

# ## Importing Libraries

# In[164]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from pandas_profiling import ProfileReport


# In[165]:


##import/integrate data set in a varaible


# In[166]:


df = pd.read_excel("default_clients2.xlsx")


# In[167]:


df.columns


# In[168]:


df.head()


# ### Hyphothesis 1 - Customer, whose has high bill may lead to late payments?

# In[172]:


df.profile_report()


# In[123]:


df.shape


# In[124]:


#Customer Age Bins
bins = [0,25,43,54,85]

#Customer Group Names
gen = ['Gen','Mel','GeX','Bmr']


# In[125]:


df['AgeBin']= pd.cut(df.AGE, bins, labels = gen)


# In[126]:


df.EDUCATION.unique()


# In[127]:


#Credt Limit Age Bins
bins = [0,50000,100000,200000,500000,750000,1000000]

creditlimit = ['Limit50k','Limit100k','Limit200k','Limit500k','Limit750k','Limit1mil']

df['CreditLimitBin']= pd.cut(df.LIMIT_BAL, bins, labels = creditlimit)


# In[128]:


df.columns


# In[129]:


#Selct Columns to DF1
columns_drop = ['LIMIT_BAL','MARRIAGE', 'AGE']
df1 = df.drop(columns_drop, axis=1)


# In[130]:


df1.info()


# In[131]:


#ProfileReport(df1)


# In[132]:


#Lable Encoding
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le = LabelEncoder()
le.fit(df1['Y'])

le1 = LabelEncoder()
le1.fit(df1['AgeBin'])

le2 = LabelEncoder()
le2.fit(df1['CreditLimitBin'])

le3 = LabelEncoder()
le3.fit(df1['SEX'])

le4 = LabelEncoder()
le4.fit(df1['EDUCATION'])


# In[133]:


df1['Y'] = le.transform(df1['Y'])
df1['AgeBin'] = le1.transform(df1['AgeBin'])
df1['CreditLimitBin'] = le2.transform(df1['CreditLimitBin'])
df1['SEX'] = le3.transform(df1['SEX'])
df1['EDUCATION'] = le4.transform(df1['EDUCATION'])


# In[134]:


df1.info()


# In[135]:


df.groupby('CreditLimitBin')['AGE'].mean(),df.groupby('CreditLimitBin')['AGE'].count(),


# In[136]:


df1.groupby('AgeBin')['Y'].mean(),df1.groupby('AgeBin')['Y'].count(),


# In[137]:


df1.info()


# In[138]:


df1.iloc[lambda x: x.index % 2 == 0]


# In[139]:


X = df1.loc[:, df1.columns != 'Y']
y = df1.iloc[:,-3]


# In[140]:


print(y)


# In[141]:


#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[142]:


##from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_dt.fit(X_train, y_train)


# In[146]:


y_pred_dt = classifier_dt.predict(X_test)


# In[147]:


print(y_pred_dt)


# In[148]:


print(classification_report(y_test, y_pred_dt))


# In[149]:


from sklearn.metrics import confusion_matrix
import pylab as pl

cm = confusion_matrix(y_test, y_pred_dt)
pl.matshow(cm)
pl.title('Confusion matrix of the classifier')
pl.colorbar()
pl.show()


# In[150]:


#Fitting random forest classification to training set 
from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy' , random_state = 0)
classifier.fit(X_train, y_train)


# In[151]:


y_predrandom= classifier.predict(X_test)


# In[152]:


print(classification_report(y_test, y_predrandom))


# In[154]:


from sklearn.metrics import confusion_matrix
import pylab as pl

cm = confusion_matrix(y_test, y_predrandom)
print(cm)


# In[43]:


from sklearn.linear_model import LogisticRegression

classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(X_train, y_train)

y_pred_logr= classifier_lr.predict(X_test)


# In[45]:


print(classification_report(y_test, y_pred_logr))


# In[55]:


df2 = df.astype('category')


# In[56]:


X = df2.loc[:, df2.columns != 'Y']
y = df2.iloc[:,-3]


# In[60]:


print(y)


# In[61]:


X1 = pd.get_dummies(X)


# In[58]:


#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.20, random_state = 0)


# In[59]:


from sklearn.linear_model import LogisticRegression

classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(X_train, y_train)

y_pred_logr= classifier_lr.predict(X_test)


# In[62]:


print(classification_report(y_test, y_pred_logr))


# # New Data Set

# In[82]:


Z = pd.read_excel("default_clients2.xlsx")


# In[83]:


Z.info()


# In[84]:


#new Data Set Customer Age Bins
bins = [0,25,43,54,85]

#Customer Group Names
gen = ['Gen','Mel','GeX','Bmr']


# In[85]:


Z['AgeBin']= pd.cut(Z.AGE, bins, labels = gen)


# In[86]:


#new Data Set Credt Limit Age Bins
bins = [0,50000,100000,200000,500000,750000,1000000]

creditlimit = ['Limit50k','Limit100k','Limit200k','Limit500k','Limit750k','Limit1mil']

Z['CreditLimitBin']= pd.cut(Z.LIMIT_BAL, bins, labels = creditlimit)


# In[89]:


Z.head()


# In[110]:


Z1 = Z.loc[:, Z.columns != 'Y']


# In[111]:


Z1.shape


# In[112]:


Z2 = Z1.astype('category')


# In[116]:


Z2.info()


# In[115]:


Z_pred_dt = classifier_lr.predict(Z2)
Z_pred_dt = Z_pred_dt.astype('str')


# In[102]:


print(Z_pred_dt)


# In[ ]:




