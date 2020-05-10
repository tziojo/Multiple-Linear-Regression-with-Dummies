#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression with Dummies - Exercise

# You are given a real estate dataset. 
# 
# Real estate is one of those examples that every regression course goes through as it is extremely easy to understand and there is a (almost always) certain causal relationship to be found.
# 
# The data is located in the file: 'real_estate_price_size_year_view.csv'. 
# 
# You are expected to create a multiple linear regression,using the new data. 
# 
# #### Regarding the 'view' variable:
# There are two options: 'Sea view' and 'No sea view'. You are expected to create a dummy variable for view and include it in the regression
# 
# Good luck!

# ## Import the relevant libraries

# In[17]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib as plt
import seaborn as sns
sns.set()


# ## Load the data

# In[18]:


data=pd.read_csv('real_estate_price_size_year_view.csv')


# In[19]:


data


# In[20]:


data.describe(include='all')


# ## Create a dummy variable for 'view'

# In[21]:


data1=data.copy()
data1['view'] = data1['view'].map({'Sea view': 1, 'No sea view': 0})


# In[22]:


data1


# ## Create the regression

# ### Declare the dependent and the independent variables

# In[23]:


y=data1['price']
x1=data1[['size','year','view']]


# ### Regression

# In[24]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# In[ ]:




