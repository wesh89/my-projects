#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm


# In[2]:


boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df = pd.read_csv(boston_url)


# In[3]:


boston_df.describe()


# In[6]:


boston_df.head(10)


# In[7]:


#Question 1: For the 'Median value of owner-occupied homes' provide a boxplot
ax = sns.boxplot(y = 'MEDV', data = boston_df)
ax.set_title('Owner-occupied homes')


# In[8]:


#Question 2: Provide a histogram for the Charles River variable
ax2 = sns.countplot(x = 'CHAS', data = boston_df)
ax2.set_title('Number of homes near the Charles River')


# In[10]:


#Question 3: Provide a boxplot for the MEDV variable vs the AGE variable - Discretize the age variable into three groups of 35 years and younger, between 35 and 50 years and older
boston_df.loc[(boston_df['AGE'] <= 35), 'Age_Group'] = '35 years and younger'
boston_df.loc[(boston_df['AGE'] > 35) & (boston_df['AGE'] < 70), 'Age_Group'] = 'between 35 and 70 years'
boston_df.loc[(boston_df['AGE'] >= 70), 'Age_Group'] = '70 years and older'


# In[11]:


ax2 = sns.boxplot(x = 'MEDV', y = 'Age_Group', data = boston_df)
ax2.set_title('Median value of owner-occupied homes per Age Group')

#Explanation
#The boxplot above shows that on average the median value of owner occupied homes is higher when the Age is lower


# In[12]:


#Question 4: Provide a scatter plot to show the relationship between Nitric oxide concentrations and the proportion of non-retail business acres per town. What can you say about the relationship?
ax3 = sns.scatterplot(y = 'NOX', x = 'INDUS', data = boston_df)
ax3.set_title('Nitric oxide concentration per proportion of non-retail business acres per town')

#Explanation
#Values in the bottom-left section of the scatter plot indicates a strong relation between low Nitric oxide concentration and low proportion of non-retail business acres per town.
#Generally, a higher proprtion of non-retail business acres per town produces a higher concentration of Nitric oxide.


# In[13]:


#Question 5: Create a histogram for the pupil to teacher ratio variable
ax4 = sns.countplot(x = 'PTRATIO', data = boston_df)
ax4.set_title('Pupil to teacher ratio per town')


# In[14]:


#Question 6: Is there a significant difference in median value of houses bounded by the Charles river or not? (T-test for independent samples)
boston_df.loc[(boston_df['CHAS'] == 0), 'CHAS_T'] = 'FAR'
boston_df.loc[(boston_df['CHAS'] == 1), 'CHAS_T'] = 'NEAR'
boston_df.head(5)

#Explanation
#Given the p-value is less than 0.05, we reject the Null Hypothesis, 
#meaning there is not a statistical difference in median value betwenn houses near the Charles River and houses far away


# In[15]:


scipy.stats.ttest_ind(boston_df[boston_df['CHAS_T'] == 'FAR']['MEDV'], 
                      boston_df[boston_df['CHAS_T'] == 'NEAR']['MEDV'], equal_var = True)


# In[17]:


#Question 7: Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)? (ANOVA)
from statsmodels.formula.api import ols
lm = ols('MEDV ~ AGE', data = boston_df).fit()
table = sm.stats.anova_lm(lm)
print(table)

#Explanation
#Given p-value is less than 0.05, we fail to accept the Null Hypothesis 
#There is statistical difference in Median values of houses (MEDV) for each proportion of owner occpied units built prior to 1940


# In[18]:


#Question 8: Can we conclude that there is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town? (Pearson Correlation)
scipy.stats.pearsonr(boston_df['NOX'], boston_df['INDUS'])

#Explanation
#Given the Pearson Coefficient is 0.76365 and p-value less than 0.05, we reject the Null Hypothesis as there is a positive correlation between Nitric oxide concentration and proportion of non-retail business acres per town
#The positive relationship is confirmed also with the Scatter Plot (Question 4)


# In[19]:


#Question 9: What is the impact of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes? (Regression analysis)
x = boston_df['DIS']
y = boston_df['MEDV']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predisction = model.predict(x)

model.summary()

#Explanation
#The coef DIS of 1.0916 indicates that an additional weighted distance to the 5 empolyment centers in boston increases of 1.0916 the median value of owner occupied homes


# In[ ]:




