#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[7]:


df = pd.read_csv('C:\\Users\\v.omsai\\Downloads\\twitter_training.csv',names = ['ID','Entity','Sentiment','Message'])
df.head()


# In[8]:


df = df.drop(columns=['ID'])


# In[9]:


df


# In[20]:


df_sentiment = df['Sentiment'].value_counts()


# In[12]:


df.isnull().sum()


# In[13]:


df = df.dropna()


# In[21]:


plt.pie(df_sentiment, labels = df_sentiment.index, autopct = '%1.1f%%')
plt.show()


# In[38]:


top5_entities = df['Entity'].value_counts(ascending=False)[:5]
top5_entities


# In[110]:


Sentiment_count = df.groupby(['Entity','Sentiment']).size().unstack(fill_value=0)
entity_sum = Sentiment_count.groupby('Entity').sum()
top5_index = top5_entities.index
entity_sum.loc[top5_index].plot(kind='bar')
plt.title('Sentiment Counts of Top 5 Entities')
plt.xlabel('Entity Names')
plt.ylabel('Frequency')
plt.show()


# In[95]:


Google_data = df[df['Entity']=='Google']
Google_sentiments_counts = Google_data['Sentiment'].value_counts()
Google_sentiments_counts.plot(kind = 'bar',color = ['red','orange','green','yellow'])
plt.title('Sentiment Counts of Google')
plt.xlabel('Sentiments')
plt.ylabel('Frequency')
plt.show()


# In[115]:


entity_microsoft = df[df['Entity']=='Microsoft']
Sentiment_count_microsoft = entity_microsoft['Sentiment'].value_counts()
Sentiment_count_microsoft.plot(kind='bar',color=['green','blue','purple','red'])


# In[ ]:




