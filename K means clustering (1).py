#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from matplotlib import style
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# In[4]:


wine_df = pd.read_csv('winequality-red.csv', sep=';')
wine_df.head()


# In[5]:


wine_df.shape


# In[6]:


wine_df.info()


# In[7]:


wine_df.isnull().sum()


# In[8]:


wine_df.describe()


# In[9]:


plt.figure(figsize=(10,7))
sns.heatmap(wine_df.corr(), annot=True)
plt.title('Correlation between the columns')
plt.show()


# In[10]:


wine_df = wine_df.drop('quality', axis=1)
wine_df.columns


# In[11]:


wss=[]
for i in range(1,10):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(wine_df)
    wss.append(kmeans.inertia_)
    
plt.plot(range(1,10), wss)
plt.title('The Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.show()


# In[13]:


pip install yellowbrick


# In[14]:


from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10), timings = False)
visualizer.fit(wine_df)
visualizer.show()


# In[16]:


for i in range(2,10):
    kmeans = KMeans(n_clusters=i, max_iter=100)
    kmeans.fit(wine_df)
    score = silhouette_score(wine_df, kmeans.labels_)
    print("For cluster: {}, the silhouette score is: {}".format(i,score))


# In[17]:


silhouette_coefficients = []
for i in range(2,10):
    kmeans = KMeans(n_clusters=i, max_iter=100)
    kmeans.fit(wine_df)
    score = silhouette_score(wine_df, kmeans.labels_)
    silhouette_coefficients.append(score)
    
plt.plot(range(2,10), silhouette_coefficients)
plt.xticks(range(2,10))
plt.xlabel("number of clusters")
plt.ylabel("Silhouette coefficient")
plt.show()


# In[18]:


pca = PCA()
X = pca.fit_transform(wine_df)


# In[21]:


kmeans = KMeans(n_clusters=4)
label = kmeans.fit_predict(X)
unique_labels = np.unique(label)


# In[23]:


for i in unique_labels:
    plt.scatter(X[label==i,0], X[label==i,1], label=i, s=20)
    
plt.legend()
plt.title('wine groups')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




