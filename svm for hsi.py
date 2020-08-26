#!/usr/bin/env python
# coding: utf-8

# In[26]:



from scipy.io import loadmat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.axes_style('whitegrid');
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


x = loadmat('PaviaU.mat')['paviaU']
y = loadmat('PaviaU_gt.mat')['paviaU_gt']
print(x.shape)


# In[27]:


fig = plt.figure(figsize = (12, 6))


q = np.random.randint(x.shape[2])
plt.imshow(x[:,:,q], cmap='nipy_spectral')
plt.axis('off')
plt.title(f'Band - {q}')
plt.savefig('IP_Bands.png')


# In[28]:


plt.figure(figsize=(12,6))
plt.imshow(y,cmap='jet')
plt.colorbar()
plt.axis('off')
plt.show()


# In[29]:




def extract_pixels(x, y):
  q = x.reshape(-1, x.shape[2])
  df = pd.DataFrame(data = q)
  df = pd.concat([df, pd.DataFrame(data = y.ravel())], axis=1)
  df.columns= [f'band{i}' for i in range(1, 1+x.shape[2])]+['class']
  df.to_csv('hsi.csv')
  return df
  
df = extract_pixels(x, y)


# In[30]:


df2 = pd.read_csv('hsi.csv')
del df2['Unnamed: 0']
df2['class']


# In[31]:


df2.iloc[:,:-1].describe()


# In[32]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 40)
dt = pca.fit_transform(df2.iloc[:, :-1].values)
r = pd.concat([pd.DataFrame(data = dt), pd.DataFrame(data = y.ravel())], axis = 1)
r.columns = [f'PC-{i}' for i in range(1,41)]+['class']
r.head()
r.to_csv('hsi_after_pca.csv')


# In[33]:


df4 = pd.read_csv('hsi_after_pca.csv')
del df4['Unnamed: 0']



# In[34]:


fig = plt.figure(figsize = (12, 6))


q = np.random.randint(x.shape[2])
plt.imshow(x[:,:,q], cmap='nipy_spectral')
plt.axis('off')
plt.title(f'Band - {q}')
plt.savefig('IP_Bands.png')


# In[35]:


f = r[r['class'] != 0]

X = f.iloc[:, :-1].values

y = f.loc[:, 'class'].values 

names = ['Asphalt','Meadows','Gravel','Trees','Painted metal sheets','Bare Soil','Bitumen','Self-Blocking Bricks','Shadows']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11, stratify=y)

svm =  SVC(C = 100, kernel = 'rbf', cache_size = 10*1024)

svm.fit(X_train, y_train)

ypred = svm.predict(X_test)


# In[36]:


data = confusion_matrix(y_test, ypred)
df_cm = pd.DataFrame(data, columns=np.unique(names), index = np.unique(names))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (12,6))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, cmap="Reds", annot=True,annot_kws={"size": 16}, fmt='d')
plt.savefig('cmap.png', dpi=300)


# In[37]:


print(classification_report(y_test, ypred, target_names = names))


# In[38]:


l=[]
for i in range(r.shape[0]):
  if r.iloc[i, -1] == 0:
    l.append(0)
  else:
    l.append(svm.predict(r.iloc[i, :-1].values.reshape(1, -1)))


# In[39]:


clmap = np.array(l).reshape(610, 340,).astype('float')
plt.figure(figsize=(12, 6))
plt.imshow(clmap, cmap='nipy_spectral')
plt.colorbar()
plt.axis('off')
plt.savefig('IP_cmap.png')
plt.show()


# In[ ]:





# In[59]:


from sklearn.tree import DecisionTreeClassifier
f = r[r['class'] != 0]

X = f.iloc[:, :-1].values

y = f.loc[:, 'class'].values 
names = ['Asphalt','Meadows','Gravel','Trees','Painted metal sheets','Bare Soil','Bitumen','Self-Blocking Bricks','Shadows']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11, stratify=y)

model = DecisionTreeClassifier()


model = model.fit(X_train,y_train)


y_pred = model.predict(X_test)



# In[41]:


data = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(data, columns=np.unique(names), index = np.unique(names))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (12,6))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, cmap="Reds", annot=True,annot_kws={"size": 16}, fmt='d')
plt.savefig('cmap2.png', dpi=300)


# In[42]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[43]:


l2=[]
for i in range(r.shape[0]):
  if r.iloc[i, -1] == 0:
    l2.append(0)
  else:
    l2.append(model.predict(r.iloc[i, :-1].values.reshape(1, -1)))
cl2map = np.array(l2).reshape(610, 340,).astype('float')
plt.figure(figsize=(12, 6))
plt.imshow(cl2map, cmap='nipy_spectral')
plt.colorbar()
plt.axis('off')
plt.savefig('IP_cmap2.png')
plt.show()


# In[44]:


print(classification_report(y_test, y_pred, target_names = names))


# In[45]:


from sklearn.neighbors import KNeighborsClassifier
f = r[r['class'] != 0]

X = f.iloc[:, :-1].values

y = f.loc[:, 'class'].values 
names = ['Asphalt','Meadows','Gravel','Trees','Painted metal sheets','Bare Soil','Bitumen','Self-Blocking Bricks','Shadows']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11, stratify=y)


model = KNeighborsClassifier(n_neighbors=5)

model = model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[46]:


data = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(data, columns=np.unique(names), index = np.unique(names))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (12,6))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, cmap="Reds", annot=True,annot_kws={"size": 16}, fmt='d')
plt.savefig('cmap5.png', dpi=300)


# In[47]:


l3=[]
for i in range(r.shape[0]):
  if r.iloc[i, -1] == 0:
    l3.append(0)
  else:
    l3.append(model.predict(r.iloc[i, :-1].values.reshape(1, -1)))
cl3map = np.array(l2).reshape(610, 340,).astype('float')
plt.figure(figsize=(12, 6))
plt.imshow(cl3map, cmap='nipy_spectral')
plt.colorbar()
plt.axis('off')
plt.savefig('IP_cmap3.png')
plt.show()


# In[61]:


print(classification_report(y_test, y_pred, target_names = names))


# In[48]:


from sklearn.ensemble import RandomForestClassifier
f = r[r['class'] != 0]

X = f.iloc[:, :-1].values

y = f.loc[:, 'class'].values 
names = ['Asphalt','Meadows','Gravel','Trees','Painted metal sheets','Bare Soil','Bitumen','Self-Blocking Bricks','Shadows']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11, stratify=y)



model3=RandomForestClassifier(n_estimators=100)

model3.fit(X_train,y_train)

y_pred=model3.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[49]:


data = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(data, columns=np.unique(names), index = np.unique(names))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (12,6))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, cmap="Reds", annot=True,annot_kws={"size": 16}, fmt='d')
plt.savefig('cmap6.png', dpi=300)


# In[50]:


l4=[]
for i in range(r.shape[0]):
  if r.iloc[i, -1] == 0:
    l4.append(0)
  else:
    l4.append(model3.predict(r.iloc[i, :-1].values.reshape(1, -1)))
cl4map = np.array(l2).reshape(610, 340).astype('float')
plt.figure(figsize=(12, 6))
plt.imshow(cl4map, cmap='nipy_spectral')
plt.colorbar()
plt.axis('off')
plt.savefig('IP_cmap4.png')
plt.show()


# In[62]:


print(classification_report(y_test, y_pred, target_names = names))


# In[ ]:




