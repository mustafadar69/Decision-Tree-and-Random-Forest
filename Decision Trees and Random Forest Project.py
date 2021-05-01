#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Random Forest Project 
# 
# For this project we will be exploring publicly available data from [LendingClub.com](www.lendingclub.com). Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help predict this.
# 
# Lending club had a [very interesting year in 2016](https://en.wikipedia.org/wiki/Lending_Club#2016), so let's check out some of their data and keep the context in mind. This data is from before they even went public.
# 
# We will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full. You can download the data from [here](https://www.lendingclub.com/info/download-data.action) or just use the csv already provided. It's recommended you use the csv provided as it has been cleaned of NA values.
# 
# Here are what the columns represent:
# * credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# * purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# * int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# * installment: The monthly installments owed by the borrower if the loan is funded.
# * log.annual.inc: The natural log of the self-reported annual income of the borrower.
# * dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# * fico: The FICO credit score of the borrower.
# * days.with.cr.line: The number of days the borrower has had a credit line.
# * revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# * revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# * inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# * delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# * pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# 

# In[2]:


loans=pd.read_csv('loan_data.csv')


# 

# In[3]:


loans.info()


# In[5]:


loans.head(5)


# In[6]:





# 

# In[12]:


plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                    bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='blue',
                                    bins=30,label='Credit.Policy=0')

plt.legend()


# ** Create a similar figure, except this time select by the not.fully.paid column.**

# In[16]:


plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                    bins=30,label='not fully paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='blue',
                                    bins=30,label='not fully paid=0')

plt.legend()


# 

# In[18]:


sns.countplot(x='purpose',hue='not.fully.paid',data=loans)


# 

# In[19]:


sns.jointplot(x='fico',y='int.rate',data=loans)


# 

# In[21]:


sns.lmplot(x ='fico', y ='int.rate',
           fit_reg = True, data = loans,hue='credit.policy',col='not.fully.paid')


# 

# In[22]:


loans.info()


# 

# In[27]:


cat_feats=['purpose']


# 

# In[28]:


final_data=pd.get_dummies(loans,columns=cat_feats,drop_first=True)


# In[30]:


final_data.info()


# 

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# 

# In[34]:


from sklearn.tree import DecisionTreeClassifier


# 

# In[35]:


dtree=DecisionTreeClassifier()


# In[36]:


dtree.fit(X_train,y_train)


# 

# In[37]:


prediction=dtree.predict(X_test)


# In[39]:


from sklearn.metrics import classification_report,confusion_matrix


# In[40]:


print(classification_report(y_test,prediction))


# In[42]:


print(confusion_matrix(y_test,prediction))


# 

# In[44]:


from sklearn.ensemble import RandomForestClassifier


# In[46]:


x=RandomForestClassifier(n_estimators=600)


# 

# In[49]:


x.fit(X_train,y_train)
pred_i=x.predict(X_test)


# 

# In[53]:


print(classification_report(y_test,pred_i))


# In[30]:





# 

# In[54]:


print(confusion_matrix(y_test,pred_i))


# 

# In[36]:





# 
