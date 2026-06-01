#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[4]:


data = pd.read_csv("creditcard.csv")

data.head()


# In[5]:


data.shape


# In[6]:


data.isnull().sum()


# In[7]:


data['Class'].value_counts().plot(kind = "bar")

plt.xlabel('class')
plt.ylabel('count')
plt.title('normal vs fraud detection')
plt.show()


# In[8]:


data['Time'].plot(kind='hist')

plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Time vs Frequency histogram')
plt.show()


# In[9]:


data['Amount'].plot(kind='hist', bins=100)

plt.xlim(0, 3000)

plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.title("Transaction Amount Distribution (Zoomed)")

plt.show()


# In[10]:


data['Class'].value_counts()


# In[11]:


#Remove duplicate data
data.duplicated().sum()


# In[12]:


data = data.drop_duplicates()
print("Duplacates have removed successfully.")

print(f"After removiing duplicates the data is:{data.duplicated().sum()}")


# In[13]:


fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]

print(fraud['Amount'].describe())
print(normal['Amount'].describe())


# In[14]:


#Scaling both Amount and Time.
scaler = StandardScaler()

data['scaled_amount'] = scaler.fit_transform(data[['Amount']])
data['scaled_time'] = scaler.fit_transform(data[['Time']]) 

data['scaled_amount']


# In[15]:


data = data.drop(['Amount', 'Time'], axis = 1)


# In[16]:


selected_features = [
    'V1',
    'V2',
    'V3',
    'V4',
    'V10',
    'V12',
    'V14',
    'scaled_amount',
    'scaled_time'
]

x = data[selected_features]
y = data['Class']


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# In[18]:


model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight='balanced',
    random_state=42
)

model.fit(x_train, y_train)


# In[19]:


y_pred = model.predict(x_test)


# In[20]:


accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# In[21]:


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[22]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[24]:


sample_data = x_test.head(20)

sample_data.to_csv("sample_transactions.csv", index=False)


# In[ ]:




