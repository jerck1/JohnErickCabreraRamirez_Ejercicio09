#!/usr/bin/env python
# coding: utf-8

# In[81]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
#estandarizacion:
import sklearn.preprocessing 
from numpy import linalg as LA
get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


data = pd.read_csv('USArrests.csv')
data2 = pd.read_csv('Cars93.csv')


# In[94]:


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


# In[95]:


y2 = np.array(data2['Price'])
columns = ['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile', 
          'Fuel.tank.capacity', 'Length', 'Width', 'Turn.circle', 'Weight']
x2 = np.array(data2[columns])
#print(x2)
print(np.shape(x2))


# In[99]:


sig1=np.cov(x)
sig2=np.cov(x2)


# In[100]:


w1, v1=LA.eig(sig1)
w2, v2=LA.eig(sig2)


# In[117]:


print(w1[0:4])


# In[116]:


pca_v1=np.ones((50,2))
pca_v1[:,0]=v1[0]
pca_v1[:,1]=v1[1]
pca_v2=np.ones((93,2))
pca_v2[:,0]=v2[0]
pca_v2[:,1]=v2[1]


# In[ ]:




