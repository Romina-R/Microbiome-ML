#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[2]:


# mergedf = pd.read_csv('metadata_otu_merged_famID-to-str.csv')
mergedf = pd.read_csv('metadata_otu_merged_famID-to-str_onlyFamwPet.csv')
mergedf.head(1)


# In[3]:


print(mergedf.shape)
mergedf = mergedf.drop("Unnamed: 0", axis=1)
mergedf = mergedf[mergedf.family_relationship !="none"]
print(mergedf.shape)


# In[4]:


human_data_train = mergedf[mergedf.host_common_name =="human"]
dog_data_test = mergedf[mergedf.host_common_name =="dog"]

X_train = human_data_train.iloc[:, 53:1085]
X_test = dog_data_test.iloc[:, 53:1085]
y_train = human_data_train["familyID"].values.reshape(-1, 1)
y_test = dog_data_test["familyID"].values.reshape(-1, 1)


# In[5]:


from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler().fit(X_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# In[6]:


# from scaled data
from sklearn.ensemble import RandomForestClassifier
rf_scaled = RandomForestClassifier(n_estimators=200)
rf_scaled = rf_scaled.fit(X_train_scaled, y_train.ravel())
rf_scaled.score(X_test_scaled, y_test) #geni impurity coefficient


# # PLAYING WITH FILTERING

# In[7]:


# mergedf_filter = mergedf#[mergedf.role =="Partner"]
# mergedf_filter = mergedf_filter[mergedf_filter.sample_type !="stool"]

# X_f = mergedf_filter.iloc[:, 53:1085]
# y_f = mergedf_filter["familyID"].values.reshape(-1, 1)
# print(X_f.shape, y_f.shape)


# In[8]:


# human_data_train = mergedf_filter[mergedf_filter.host_common_name =="human"]
# dog_data_test = mergedf_filter[mergedf_filter.host_common_name =="dog"]

# X_train = human_data_train.iloc[:, 53:1085]
# X_test = dog_data_test.iloc[:, 53:1085]
# y_train = human_data_train["familyID"].values.reshape(-1, 1)
# y_test = dog_data_test["familyID"].values.reshape(-1, 1)


# In[9]:


# rf_f = RandomForestClassifier(n_estimators=200)
# rf_f = rf_f.fit(X_train, y_train.ravel())
# rf_f.score(X_test, y_test) #geni impurity coefficient


# In[10]:


# #only looking at human families, predict which family each person belongs to
# human_data = mergedf_filter[mergedf_filter.family_relationship !="none"]
# human_data = mergedf_filter[mergedf_filter.host_common_name =="human"]
# human_data = mergedf_filter[mergedf_filter.role !="offspring"]

# X = mergedf.iloc[:, 53:1085]
# y = mergedf["familyID"].values.reshape(-1, 1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# rf = RandomForestClassifier(n_estimators=200)
# rf = rf.fit(X_train, y_train.ravel())
# rf.score(X_test, y_test) #geni impurity coefficient


# In[ ]:




