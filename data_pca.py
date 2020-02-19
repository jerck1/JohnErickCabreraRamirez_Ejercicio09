#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
#estandarizacion:
import sklearn.preprocessing 
from numpy import linalg as LA
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('USArrests.csv')
data2 = pd.read_csv('Cars93.csv')


# In[3]:


#print(np.shape(data))
#print(data[1:3])
columns=data.keys()
columns=['Murder', 'Assault', 'UrbanPop', 'Rape']
x=np.array(data[columns])#[:,1:]
states=np.array(data[columns])[:,0]
#print(np.concatenate((x[:,0],x[:,0])))
#print(x)
#x[:,0]
#xx=np.zeros((len(x),4))
#for i in range(len(x)):
 #   for j in range(4):
  #      xx[i,j]=x[i,j]
#print(x)
#x


# In[4]:


y2 = np.array(data2['Price'])
columns = ['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile', 
          'Fuel.tank.capacity', 'Length', 'Width', 'Turn.circle', 'Weight']
x2 = np.array(data2[columns])
#print(x2)
print(np.shape(x2))


# In[5]:


sig1=np.cov(x)
sig2=np.cov(x2)


# In[6]:


w1, v1=LA.eig(sig1)
w2, v2=LA.eig(sig2)


# In[7]:


print(w1[0:4])


# In[8]:


pca_v1=np.ones((2,4))
pca_v2=np.ones((2,4))
for i in range(2):
    pca_v1[i]=v1[i,:4]/LA.norm(v1[i,:4])
    pca_v2[i]=v2[i,:4]/LA.norm(v2[i,:4])
#pca_v1[:,0]
#v1[0,:4]


# In[16]:


plt.plot()
#plt.scatter(X, Y)
#plt.scatter(X_scaled[:,3], lasso.predict(X_scaled), marker='^')
#plt.xlabel(columns[3])
#plt.ylabel('Price')
#plt.quiver([0, 0], pca_v1[:,0])
#plt.quiver([0, 0], 30)
#plt.arrow([0,0],pca_v1[:,0])
#plt.arrow((0,0),(1,1))


# In[10]:


0.53**2+0.58**2+0.27**2+0.54**2


# In[11]:


pca_v1[0,0]


# In[12]:


pca_v1[:,0]


# In[ ]:




