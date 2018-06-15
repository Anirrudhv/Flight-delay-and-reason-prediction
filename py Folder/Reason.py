
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import csv


# In[2]:


df = pd.read_csv('TIME1.csv')
df.shape


# In[3]:


df.shape
X = df.loc[df['ARR_DELAY_NEW']!=0]
X


# In[4]:


pd.set_option('display.max_columns', None)


# In[5]:


X_Carr=X.drop(['WEATHER_DELAY',
 'NAS_DELAY',
 'SECURITY_DELAY',
 'LATE_AIRCRAFT_DELAY'],axis=1)


# In[6]:


X_weather=X.drop(['NAS_DELAY',
 'SECURITY_DELAY',
 'LATE_AIRCRAFT_DELAY','CARRIER_DELAY'],axis=1)


# In[7]:


X_NAs=X.drop(['CARRIER_DELAY','WEATHER_DELAY','SECURITY_DELAY',
 'LATE_AIRCRAFT_DELAY'],axis=1)


# In[8]:


X_late=X.drop(['CARRIER_DELAY','WEATHER_DELAY',
 'NAS_DELAY',
 'SECURITY_DELAY'],axis=1)


# In[9]:


X_Security=X.drop(['CARRIER_DELAY','WEATHER_DELAY','LATE_AIRCRAFT_DELAY','CARRIER_DELAY'],axis=1)


# In[10]:


X_Carr.corr()


# In[11]:


X_weather.corr()


# In[12]:


X_NAs.corr()


# In[13]:


X_late.corr()


# In[14]:


X_Security.corr()


# In[15]:


X.describe()


# In[16]:


X_correct=X.loc[df['ARR_DELAY'] < 100]
X_correct


# In[17]:


X_Carr=X_correct[['DEP_DELAY','DEP_DEL15','DAY_OF_WEEK','DEP_DELAY_NEW','ARR_DELAY_GROUP','CARRIER_DELAY']]

X_Carr.describe()


# In[18]:


X_Carr['ARR_DELAY_GROUP']=X_Carr['ARR_DELAY_GROUP'].fillna(8)
X_Carr['DEP_DELAY']=X_Carr['DEP_DELAY'].fillna(20.18)
X_Carr['DEP_DEL15']=X_Carr['DEP_DEL15'].fillna(0.46)
X_Carr['DAY_OF_WEEK']=X_Carr['DAY_OF_WEEK'].fillna(3.55)
X_Carr['DEP_DELAY_NEW']=X_Carr['DEP_DELAY_NEW'].fillna(21.45)
X_Carr['CARRIER_DELAY']=X_Carr['CARRIER_DELAY'].fillna(999)


# In[19]:


X_Carr.shape


# In[20]:


X_weather = X_correct[['DEP_DELAY_NEW','DEP_DELAY_GROUP','ARR_DELAY_NEW','ARR_DELAY_GROUP','WEATHER_DELAY']]


# In[21]:


X_weather.describe()


# In[22]:


X_weather['ARR_DELAY_GROUP']=X_weather['ARR_DELAY_GROUP'].fillna(8)
X_weather['DEP_DELAY_NEW']=X_weather['DEP_DELAY_NEW'].fillna(21.4)
X_weather['DEP_DELAY_GROUP']=X_weather['DEP_DELAY_GROUP'].fillna(13)
X_weather['ARR_DELAY_NEW']=X_weather['ARR_DELAY_NEW'].fillna(22.7)
X_weather['WEATHER_DELAY']=X_weather['WEATHER_DELAY'].fillna(999)


# In[23]:


X_NAs = X_correct[['TAXI_OUT','ARR_DELAY_NEW','ARR_DELAY_GROUP','NAS_DELAY']]


# In[24]:


X_NAs.describe()


# In[25]:


X_NAs['TAXI_OUT']=X_NAs['TAXI_OUT'].fillna(22.26)
X_NAs['ARR_DELAY_NEW']=X_NAs['ARR_DELAY_NEW'].fillna(22.7)
X_NAs['ARR_DELAY_GROUP']=X_NAs['ARR_DELAY_GROUP'].fillna(8)
X_NAs['NAS_DELAY']=X_NAs['NAS_DELAY'].fillna(999)


# In[26]:


X_late = X_correct[['AIRLINE_ID','FL_NUM','DEP_DELAY_NEW','DEP_DEL15','DEP_DELAY_GROUP','WHEELS_OFF','ARR_DELAY','LATE_AIRCRAFT_DELAY']]


# In[27]:


X_late.describe()


# In[28]:


X_late['AIRLINE_ID']=X_late['AIRLINE_ID'].fillna(0)
X_late['FL_NUM']=X_late['FL_NUM'].fillna(0)
X_late['DEP_DELAY_NEW']=X_late['DEP_DELAY_NEW'].fillna(21.45)
X_late['DEP_DEL15']=X_late['DEP_DEL15'].fillna(0.46)
X_late['DEP_DELAY_GROUP']=X_late['DEP_DELAY_GROUP'].fillna(14)
X_late['WHEELS_OFF']=X_late['WHEELS_OFF'].fillna(1424.5)
X_late['ARR_DELAY']=X_late['ARR_DELAY'].fillna(22.7)
X_late['LATE_AIRCRAFT_DELAY']=X_late['LATE_AIRCRAFT_DELAY'].fillna(999)


# In[29]:


X_late


# In[30]:


X_Security.corr()


# In[31]:


X_Carr_train=X_Carr[X_Carr['CARRIER_DELAY'] < 999]
X_Carr_train=X_Carr_train.drop(['CARRIER_DELAY'],axis=1)
X_Carr_train


# In[32]:


X_Carr_test=X_Carr[X_Carr['CARRIER_DELAY']== 999]
X_Carr_test=X_Carr_test.drop(['CARRIER_DELAY'],axis=1)
X_Carr_test


# In[33]:


Y_Carr  = X_Carr[['CARRIER_DELAY']]


# In[34]:


Y_Carr_test=Y_Carr[Y_Carr['CARRIER_DELAY']== 999]
Y_Carr_train=Y_Carr[Y_Carr['CARRIER_DELAY']< 999]


# In[35]:


Y_Carr_test
Y_Carr_train


# In[37]:


from sklearn import linear_model, svm
from sklearn import cross_validation
X_train, X_test, y_train, y_test = X_Carr_train, X_Carr_test, Y_Carr_train,Y_Carr_test

clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)
y_Carr_res = clf.predict(X_test)


# In[38]:


X_weather_train=X_weather[X_weather['WEATHER_DELAY'] < 999]
X_weather_train=X_weather_train.drop(['WEATHER_DELAY'],axis=1)
X_weather_train


# In[39]:


X_weather_test=X_weather[X_weather['WEATHER_DELAY']== 999]
X_weather_test=X_weather_test.drop(['WEATHER_DELAY'],axis=1)
X_weather_test


# In[40]:


Y_weather  = X_weather[['WEATHER_DELAY']]


# In[41]:


Y_weather_test=Y_weather[Y_weather['WEATHER_DELAY']== 999]
Y_weather_train=Y_weather[Y_weather['WEATHER_DELAY']< 999]


# In[42]:



X_train, X_test, y_train, y_test = X_weather_train, X_weather_test, Y_weather_train,Y_weather_test

clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)
y_weather_res = clf.predict(X_test)


# In[43]:


X_NAs_train=X_NAs[X_NAs['NAS_DELAY'] < 999]
X_NAs_train=X_NAs_train.drop(['NAS_DELAY'],axis=1)
X_NAs_train


# In[44]:


X_NAs_test=X_NAs[X_NAs['NAS_DELAY']== 999]
X_NAs_test=X_NAs_test.drop(['NAS_DELAY'],axis=1)
X_NAs_test


# In[45]:


Y_NAs  = X_NAs[['NAS_DELAY']]
Y_NAs_test=Y_NAs[Y_NAs['NAS_DELAY']== 999]
Y_NAs_train=Y_NAs[Y_NAs['NAS_DELAY']< 999]


# In[46]:


from sklearn import linear_model, svm
from sklearn import cross_validation
X_train, X_test, y_train, y_test = X_NAs_train, X_NAs_test, Y_NAs_train,Y_NAs_test

clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)
y_NAS_res = clf.predict(X_test)


# In[47]:


X_late_train=X_late[X_late['LATE_AIRCRAFT_DELAY'] < 999]
X_late_train=X_late_train.drop(['LATE_AIRCRAFT_DELAY'],axis=1)
X_late_train


# In[48]:


X_late_test=X_late[X_late['LATE_AIRCRAFT_DELAY']== 999]
X_late_test=X_late_test.drop(['LATE_AIRCRAFT_DELAY'],axis=1)
X_late_test


# In[49]:


Y_late  = X_late[['LATE_AIRCRAFT_DELAY']]
Y_late_test=Y_late[Y_late['LATE_AIRCRAFT_DELAY']== 999]
Y_late_train=Y_late[Y_late['LATE_AIRCRAFT_DELAY']< 999]


# In[50]:


from sklearn import linear_model, svm
from sklearn import cross_validation
X_train, X_test, y_train, y_test = X_late_train, X_late_test, Y_late_train,Y_late_test

clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)
y_late_res = clf.predict(X_test)


# In[51]:


data=X_late_test['ARR_DELAY']


# In[52]:


from pandas import DataFrame
Y_Carr_res = DataFrame(data=y_Carr_res)
Y_Carr_res.columns = ['C']
#Y_Carr_res = Y_Carr_res.rename(columns = {"0" : "Carrier"})
#Y_Carr_res.rename(index=str, columns={"0": "C"})
Y_NAS_res = DataFrame(data=y_NAS_res)
Y_NAS_res.columns = ['N']
Y_Weather_res = DataFrame(data=y_weather_res)
Y_Weather_res.columns = ['W']
Y_late_res = DataFrame(data=y_late_res)
Y_late_res.columns = ['L']


# In[53]:



#Y_Carr_res=pd.DataFrame(data=y_Carr_res[0:,1:], index=y_Carr_res[0:,0], columns=y_Carr_res[0,1:])
#Y_NAS_res=pd.DataFrame(data=y_Carr_res[0:,1:], index=y_NAS_res[0:,0], columns=y_NAS_res[0,1:])


# In[54]:


result = pd.concat([Y_NAS_res, Y_Carr_res], axis=1)
result = pd.concat([result,Y_Weather_res],axis=1)
result = pd.concat([result,Y_late_res],axis=1)
result


# In[55]:


data1 = DataFrame(data=data)


# In[56]:


data1.shape


# In[57]:


result[result<0]=0
result


# In[58]:


data1


# In[59]:


result['New_Delay']=result['C']+result['N']+result['W']+result['L']


# In[60]:


result


# In[61]:


import numpy as np
from sklearn.metrics import accuracy_score


# In[62]:


r1= result['New_Delay']


# In[63]:


result


# In[64]:


d1 = data1['ARR_DELAY']


# In[65]:


d1.shape


# In[66]:


from sklearn.metrics import mean_squared_error
print("Mean squared error: %.2f" % mean_squared_error(d1, r1))
RSME_line = np.sqrt(mean_squared_error(d1,r1))
print ("RMSE : %.2f" %RSME_line)


# from sklearn.metrics import mean_absolute_error
# print("Mean absolute error: %.2f" % mean_absolute_error(d1, r1))

# In[67]:


import matplotlib.pyplot as plt


# In[68]:


r1_zscore = (r1-r1.mean())/r1.std()


# In[69]:


r1_zscore


# In[70]:


d1.index = range(81677)


# In[71]:


d1


# In[72]:


d1_zscore = (d1-d1.mean())/d1.std()


# In[73]:


d1_zscore


# In[74]:


r1_zscore[0]/d1_zscore[0]


# In[75]:


pop = r1_zscore/d1_zscore
pop.mean()


# In[76]:



get_ipython().run_cell_magic(u'HTML', u'', u'<h1>Ridge Regression</h1>')


# In[77]:


from sklearn.linear_model import Ridge
X_train_ridge, X_test_ridge, y_train_ridge, y_test_ridge = X_late_train, X_late_test, Y_late_train,Y_late_test
clf = Ridge(alpha=1.0)
clf.fit(X_train_ridge, y_train_ridge)


# In[78]:


y_res_ridgel = clf.predict(X_test_ridge)
print y_res_ridgel


# In[79]:


from sklearn.linear_model import Ridge
X_train_ridge, X_test_ridge, y_train_ridge, y_test_ridge = X_Carr_train, X_Carr_test, Y_Carr_train,Y_Carr_test
clf = Ridge(alpha=1.0)
clf.fit(X_train_ridge, y_train_ridge)


# In[80]:


y_res_ridgec = clf.predict(X_test_ridge)
y_res_ridgec


# In[81]:


from sklearn.linear_model import Ridge
X_train_ridge, X_test_ridge, y_train_ridge, y_test_ridge = X_NAs_train, X_NAs_test, Y_NAs_train,Y_NAs_test
clf = Ridge(alpha=1.0)
clf.fit(X_train_ridge, y_train_ridge)


# In[82]:


y_res_ridgen = clf.predict(X_test_ridge)


# In[83]:


from sklearn.linear_model import Ridge
X_train_ridge, X_test_ridge, y_train_ridge, y_test_ridge = X_weather_train, X_weather_test, Y_weather_train,Y_weather_test
clf = Ridge(alpha=1.0)
clf.fit(X_train_ridge, y_train_ridge)


# In[84]:


y_res_ridgew = clf.predict(X_test_ridge)


# In[85]:


from pandas import DataFrame
y_res_ridgel = DataFrame(data=y_res_ridgel)
y_res_ridgel.columns = ['L']
#Y_Carr_res = Y_Carr_res.rename(columns = {"0" : "Carrier"})
#Y_Carr_res.rename(index=str, columns={"0": "C"})
y_res_ridgec = DataFrame(data=y_res_ridgec)
y_res_ridgec.columns = ['C']
y_res_ridgen = DataFrame(data=y_res_ridgen)
y_res_ridgen.columns = ['N']
y_res_ridgew = DataFrame(data=y_res_ridgew)
y_res_ridgew.columns = ['W']


# In[86]:


result = pd.concat([y_res_ridgel, y_res_ridgec], axis=1)
result = pd.concat([result,y_res_ridgen],axis=1)
result = pd.concat([result,y_res_ridgew],axis=1)
result


# In[87]:


result[result<0]=0
result


# In[88]:


data=X_late_test['ARR_DELAY']


# In[89]:


data1 = DataFrame(data=data)


# In[90]:


data1


# In[91]:


d1 = data1['ARR_DELAY']


# In[92]:


d1.index = range(81677)


# In[93]:


result['New_Delay']=result['C']+result['N']+result['W']+result['L']


# In[94]:


r1 = result['New_Delay']


# In[95]:


r1_zscore = (r1-r1.mean())/r1.std()


# In[96]:


d1_zscore = (d1-d1.mean())/d1.std()


# In[97]:


pop = r1_zscore/d1_zscore
pop.mean()


# In[98]:


from sklearn.metrics import mean_squared_error
print("Mean squared error: %.2f" % mean_squared_error(d1, r1))
RSME_ridge = np.sqrt(mean_squared_error(d1,r1))
print ("RMSE : %.2f" %RSME_ridge)


# In[99]:


obj = ("Linear","Ridge")
y_pos = np.arange(len(obj))
per = [RSME_line,RSME_ridge]
plt.bar(y_pos,per,align = 'center',alpha = 0.5 )
plt.xticks(y_pos,obj)
plt.ylabel('RMSE')
plt.show()
get_ipython().run_cell_magic(u'HTML', u'', u'<h1>Logistic Regression</h1>')


# In[106]:


from sklearn import linear_model, datasets
X_train_poly, X_test_poly, y_train_poly, y_test_poly = X_weather_train, X_weather_test, Y_weather_train,Y_weather_test
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train_poly, y_train_poly)


# In[108]:


y_res_logw = logreg.predict(X_test_poly)


# In[109]:


print y_res_logw


# In[114]:


from sklearn import linear_model, datasets
X_train_poly, X_test_poly, y_train_poly, y_test_poly = X_Carr_train, X_Carr_test, Y_Carr_train,Y_Carr_test
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train_poly, y_train_poly)


# In[115]:


y_res_logc = logreg.predict(X_test_poly)


# In[116]:


from sklearn import linear_model, datasets
X_train_poly, X_test_poly, y_train_poly, y_test_poly = X_NAs_train, X_NAs_test, Y_NAs_train,Y_NAs_test
logreg = Ridge(alpha=1.0)
logreg.fit(X_train_poly, y_train_poly)


# In[121]:


y_res_logn = logreg.predict(X_test_poly)


# In[122]:


from sklearn import linear_model, datasets
X_train_poly, X_test_poly, y_train_poly, y_test_poly = X_late_train, X_late_test, Y_late_train,Y_late_test
logreg = Ridge(alpha=1.0)
logreg.fit(X_train_poly, y_train_poly)


# In[124]:


y_res_logl = logreg.predict(X_test_poly)


# In[126]:


from pandas import DataFrame
y_res_logw = DataFrame(data=y_res_logw)
y_res_logw.columns = ['W']
#Y_Carr_res = Y_Carr_res.rename(columns = {"0" : "Carrier"})
#Y_Carr_res.rename(index=str, columns={"0": "C"})
y_res_logc = DataFrame(data=y_res_logc)
y_res_logc.columns = ['C']
y_res_logn = DataFrame(data=y_res_logn)
y_res_logn.columns = ['N']
y_res_logl = DataFrame(data=y_res_logl)
y_res_logl.columns = ['L']


# In[127]:


result = pd.concat([y_res_logw, y_res_logc], axis=1)
result = pd.concat([result,y_res_logn],axis=1)
result = pd.concat([result,y_res_logw],axis=1)
result

