# CA_2- MSC_DA_BD_ADAv5

Identify and Carry out an analysis of a large a Dataset gleaned from X ( Foremerly Twitter) api
#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install pymongo')


# In[3]:


import pymongo
import pandas as pd
import json


# In[4]:


client=pymongo.MongoClient("mongodb://localhost:27017")


# In[11]:


# Load the dataset
file_path = 'ProjectTweets.csv'
df = pd.read_csv(file_path, sep=',', names=['ids', 'date', 'flag', 'user', 'text'], skiprows=1)  # Skip the first row if it's a header row


# In[12]:


df.head()


# In[13]:





# In[14]:


data=df.to_dict(orient="records")


# In[15]:


data


# In[16]:


db=client["tweet_db"]


# In[17]:


print(db)


# In[18]:


db.tweet.insert_many(data)
