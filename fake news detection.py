#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("hello world")


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[3]:


df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")


# In[4]:


df_fake.head()


# In[5]:


df_fake["class"] = 0
df_true["class"] = 1


# In[6]:


df_fake.shape, df_true.shape


# In[7]:


df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
    
    
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)


# In[8]:


# the above written program helps in deleting the last 10 datas for the manual testing


# In[9]:


df_fake.shape, df_true.shape


# In[10]:


df_fake_manual_testing.loc["class"] = 0
df_true_manual_testing.loc["class"] = 1


# In[11]:


df_fake_manual_testing.head(10)


# In[12]:


df_true_manual_testing.head(10)


# In[13]:


df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("manual_testing.csv")


# In[14]:


df_merge = pd.concat([df_fake, df_true], axis =0 )
df_merge.head(10)


# In[15]:


df_merge.columns


# In[16]:


df = df_merge.drop(["title", "subject","date"], axis = 1)


# In[17]:


df.isnull().sum()


# In[18]:


df = df.sample(frac = 1)


# In[19]:


df.head()


# In[20]:


df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)


# In[21]:


df.columns


# In[22]:


df.head()


# In[23]:


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


# In[24]:


df["text"] = df["text"].apply(wordopt)


# In[25]:


x = df["text"]
y = df["class"]


# In[26]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# In[27]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# In[28]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train,y_train)


# In[29]:


pred_lr=LR.predict(xv_test)
sns.countplot(pred_lr)


# In[30]:


LR.score(xv_test, y_test)


# In[31]:


print(classification_report(y_test, pred_lr))


# In[32]:


from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[33]:


pred_dt = DT.predict(xv_test)
sns.countplot(pred_dt)


# In[34]:


accuracy=DT.score(xv_test, y_test)
accuracy*100


# In[35]:


print(classification_report(y_test, pred_dt))


# In[36]:


print("hello")


# In[ ]:





# In[ ]:





# In[ ]:





# In[37]:


from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)


# In[38]:


pred_rfc = RFC.predict(xv_test)


# In[39]:


RFC.score(xv_test, y_test)


# In[40]:


print(classification_report(y_test, pred_rfc))


# In[41]:


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]),                                                                                                       output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0])))


# In[ ]:


news = str(input())
manual_testing(news)


# In[1]:


import requests

# Set up the API endpoint and the API key
endpoint = 'https://api.nytimes.com/svc/news/v3/content/all/all.json'
api_key = '<your API key>'

# Set up the request parameters
params = {
    'api-key': api_key,
    'limit': 10  # retrieve the 10 latest news articles
}

# Send the HTTP request to the API
response = requests.get(endpoint, params=params)

# Process the response
if response.status_code == 200:
    # Parse the response as JSON
    data = response.json()
    # Extract the articles from the response
    articles = data['results']
    # Process the articles as needed
    for article in articles:
        print(article['title'])
else:
    print(f'Request failed with status code {response.status_code}')


# In[ ]:





# In[ ]:





# In[ ]:




