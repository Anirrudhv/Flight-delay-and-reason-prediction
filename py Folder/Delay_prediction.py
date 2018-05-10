
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('TIME1.csv')


# In[3]:


df.shape


# In[4]:


pd.set_option('display.max_columns', None)
df


# In[5]:


cols = [
    'YEAR','UNIQUE_CARRIER','AIRLINE_ID','FL_NUM','ORIGIN_AIRPORT_ID','DEST','DEP_DELAY','AIR_TIME','ACTUAL_ELAPSED_TIME','DISTANCE','MONTH','DAY_OF_WEEK','FLIGHTS','ARR_DELAY' 
]


# In[6]:


X = df[cols]


# In[7]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder


# In[8]:


labelencoder_a=LabelEncoder()
X['UNIQUE_CARRIER']=labelencoder_a.fit_transform(X['UNIQUE_CARRIER'])


# In[9]:


X['DEST']=labelencoder_a.fit_transform(X['DEST'])


# In[10]:


X.corr()


# In[11]:


X.drop(['MONTH','FLIGHTS','DISTANCE','YEAR'],axis=1,inplace = True)


# In[12]:


X.drop(['ARR_DELAY'],axis=1,inplace = True)


# In[13]:


X


# In[14]:


from sklearn import linear_model, svm
from sklearn import cross_validation
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
X =  X.fillna(0)


# In[15]:


from sklearn.preprocessing import OneHotEncoder
X = X.iloc[:,:].values


# In[16]:


onehot=OneHotEncoder(categorical_features=[4])


# In[17]:


X=onehot.fit_transform(X).toarray()


# In[18]:


y = df['ARR_DELAY']


# In[19]:


y = y.fillna(0)


# In[20]:


y = y.astype(int)


# In[21]:


print X.shape


# In[22]:


print y.shape


# In[23]:



get_ipython().run_cell_magic(u'HTML', u'', u'<h1>Linear Regression</h1>')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)

#X_test = df.values

y_res = clf.predict(X_test)
print y_res


# In[24]:


print ("mean absolute error: %.2f" % mean_absolute_error(y_test, y_res))
rms_line = np.sqrt(mean_squared_error(y_test, y_res))
print(rms_line)


# In[25]:


print("Mean squared error: %.2f" % mean_squared_error(y_test, y_res))


# In[26]:


plt.plot(y_res, y_test,'ro')
plt.plot([0,1400],[0,1400],'g-')
plt.xlabel('predicted')
plt.ylabel('real')
plt.show()


# In[27]:


get_ipython().run_cell_magic(u'HTML', u'', u'<h1>Ridge Regression</h1>')


# In[28]:


from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train)


# In[29]:


y_res = clf.predict(X_test)
print y_res


# In[30]:


print ("mean absolute error: %.2f" % mean_absolute_error(y_test, y_res))
rms_ridge = np.sqrt(mean_squared_error(y_test, y_res))
print(rms_ridge)


# In[31]:


print("Mean squared error: %.2f" % mean_squared_error(y_test, y_res))


# In[32]:


get_ipython().run_cell_magic(u'HTML', u'', u'<h1>Random Forest</h1>')


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)


# In[34]:


y_res = clf.predict(X_test)
print y_res


# In[35]:


print ("mean absolute error: %.2f" % mean_absolute_error(y_test, y_res))
rms_random = np.sqrt(mean_squared_error(y_test, y_res))
print(rms_random)


# In[36]:


print("Mean squared error: %.2f" % mean_squared_error(y_test, y_res))


# In[37]:


get_ipython().run_cell_magic(u'HTML', u'', u'<h1>lasso regression</h1>')


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.linear_model import Lasso
clf = Lasso(alpha=0.00001,normalize=True, max_iter=1e5)
clf.fit(X_train, y_train)


# In[39]:


y_res = clf.predict(X_test)
print y_res


# In[40]:


print ("mean absolute error: %.2f" % mean_absolute_error(y_test, y_res))
rms_lasso = np.sqrt(mean_squared_error(y_test, y_res))
print(rms_lasso)


# In[41]:


print("Mean squared error: %.2f" % mean_squared_error(y_test, y_res))


# In[42]:


obj = ("Linear ","Ridge","RandomF","Lasso")
y_pos = np.arange(len(obj))
per = [rms_line,rms_ridge,rms_random,rms_lasso]
plt.bar(y_pos,per,align = 'center',alpha = 0.5 )
plt.xticks(y_pos,obj)
plt.ylabel('RMSE')
plt.show()

