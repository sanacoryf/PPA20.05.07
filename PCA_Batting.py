#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


fname = "Batting.csv"
df = pd.read_csv(fname)


# In[3]:


df.head()


# In[4]:


df[df["yearID"] == 1871]['AB'].median()


# In[5]:


df[df["yearID"] == 2015]['AB'].median()


# In[6]:


df[df["yearID"] == 2015]['AB'].count()


# In[7]:


df[(df["yearID"] == 2015) & (df["AB"] > 240) ]


# In[8]:


df.columns


# In[9]:


cols2keep  = ['AB', 'R', 'H','2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO']
newdf = df[(df["yearID"] > 2010) & (df["AB"] > 240) ]
newdf = newdf[cols2keep]


# In[10]:


newdf[:6]


# In[11]:


newdf['PA'] = newdf['AB']+newdf['BB']


# In[12]:


newdf[:12]


# In[13]:


statlist = ['R','H','2B','3B','HR', 'BB','SB']


for stat in statlist:
    print(stat+'perPA')
    newdf[stat+'perPA'] = newdf[stat]/newdf['PA']


# In[14]:


newdf["RperPA"].mean()


# In[15]:


newdf["RperPA"].max()


# In[16]:


newdf[newdf["RperPA"]>0.20]


# In[17]:


df[104453:104454]


# In[18]:


df[94700:94701]


# In[19]:


df[96905:96906]


# In[20]:


newdf.shape[0], newdf.dropna().shape[0]


# In[23]:


# RperPA, HperPA, 2BperPA, 3BperPA, HRperPA, SBperPA
features = ['HperPA','2BperPA','3BperPA','HRperPA', 'BBperPA','SBperPA']
x = newdf.loc[:, features].values


# In[24]:


x


# In[25]:


y = newdf.loc[:,['RperPA']].values
dfy = pd.DataFrame(y); dfy.columns = ["RperPA"]


# In[26]:


#x = StandardScaler().fit_transform(x)
pd.DataFrame(data = x, columns = features).head()


# In[27]:


pca = PCA(n_components=1)


# In[28]:


principalComponents = pca.fit_transform(x)


# In[29]:


pca.components_


# In[32]:


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
pca.components_


# In[33]:


principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, dfy['RperPA'] ], axis = 1)
finalDf[:6]


# In[34]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)



ax.scatter(finalDf['principal component 1']
               , finalDf['principal component 2'])
#ax.legend(targets)
ax.grid()


# In[35]:


#colors = ['r', 'g', 'b']


# a range of blues 
#(204,229,255), (153,204,255), (102,178,255), (51,153,255)
# (0,128,255),(0,102,204),(0,76,153),(0,51,102),(0,25,51)


numDivisions = 8

aRange = np.linspace(y.min(),y.max(),numDivisions+1)

finalDf['Rness'] = 0

for aNum in range(numDivisions):
    finalDf.loc[finalDf["RperPA"]> aRange[aNum+1],'Rness']= aNum+1
    


# In[36]:


finalDf[:6]


# In[37]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)


targets = [0,1,2,3,4,5,6,7,8]
colors = [(204,229,255), (153,204,255), (102,178,255), (51,153,255),
 (0,128,255),(0,102,204),(0,76,153),(0,51,102),(0,25,51)]

colors = [np.array(color)/255 for color in colors]
colors = [ [list(color)] for color in colors]

#colors = ['lightgray', 'silver', 'darkgray', 'gray','dimgray','black']

for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Rness'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




