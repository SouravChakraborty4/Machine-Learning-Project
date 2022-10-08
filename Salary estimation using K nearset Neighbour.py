#!/usr/bin/env python
# coding: utf-8

# In[34]:


#ML Project 2 Sourav Chakraborty


# In[49]:


import pandas as pd
import numpy as np


# In[50]:


dataset=pd.read_csv("salary.csv")

print(dataset.shape)

dataset.head(5)


# In[51]:


# mapping the salary into binary value


# In[52]:


# *Mapping Salary Data to Binary Value*"""

income_set = set(dataset['income'])
dataset['income'] = dataset['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
dataset.head(5)


# In[53]:


# *Segregate Dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable)*"""

X = dataset.iloc[:, :-1].values
X


# In[54]:


Y = dataset.iloc[:, -1].values
Y


# In[55]:


# *Splitting Dataset into Train & Test*"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[56]:


#We want our test data to be a completely new and a surprise set for our model


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)


# In[57]:


# Finding the Best K-Value

error = []
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# In[58]:


# Calculating error for K values between 1 and 40
for i in range(1, 40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    pred_i = model.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# In[59]:


# Training

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 16, metric = 'minkowski', p = 2)
model.fit(X_train, y_train)


# In[61]:


# Predicting, wheather new customer with Age & Salary will Buy or Not

age = int(input("Enter New Employee's Age: "))
edu = int(input("Enter New Employee's Education: "))
cg = int(input("Enter New Employee's Captital Gain: "))
wh = int(input("Enter New Employee's Hour's Per week: "))
newEmp = [[age,edu,cg,wh]]
result = model.predict(sc.transform(newEmp))
print(result)

if result == 1:
  print("Employee might got Salary above 50K")
else:
  print("Customer might not got  Salary above 50K")


# In[47]:


# Prediction for all Test Data

y_pred = model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[48]:


# Evaluating Model - CONFUSION MATRIX

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix: ")
print(cm)

print("Accuracy of the Model: {0}%".format(accuracy_score(y_test, y_pred)*100))


# In[ ]:




