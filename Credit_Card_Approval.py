#!/usr/bin/env python
# coding: utf-8

# In[70]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[71]:


col = ['Male', 'Age', 'Debt', 'Married', 'BankCustomer', 'EducationLevel', 'Ethnicity', 'YearsEmployed', 'PriorDefault', 'Employed', 'CreditScore', 'DriversLicense', 'Citizen', 'ZipCode', 'Income', 'Approved']
CRX_data = pd.read_csv('crx.data', names=col)


# In[72]:


# Printing data info and sample

print(CRX_data.describe(), "\n\n")
print(CRX_data.info())
CRX_data.sample(5)


# In[75]:


cols = ['Male', 'Married', 'BankCustomer', 'EducationLevel', 'Ethnicity', 'PriorDefault', 'Employed', 'DriversLicense', 'Citizen']
for col in cols:
    print('Unique values in {}: {}'.format(col, CRX_data[col].unique()))


# In[76]:


# Replacing missing values imputed as '?' to Nan and changing age and zipcode feature data type to float and int respectively

CRX_data = CRX_data.replace(['?'], np.nan)
CRX_data.Age = CRX_data.Age.astype('float')


# In[77]:


# Checking for missing values

CRX_data.isnull().sum().sort_values(ascending=False)


# In[78]:


CRX_data.fillna(CRX_data.mean(), inplace=True)  #Imputing missng values of numeric features with mean values
CRX_data.fillna('ffil', inplace=True)  # Imputing missing values of categorical feature
CRX_data.isnull().sum() # Checking null value(s)


# In[79]:


# Visualising data distribution

def plotDistPlot(col):
    """Flexibly plot a univariate distribution of observation"""
    sns.distplot(col)
    plt.show()
plotDistPlot(CRX_data['Age'])
plotDistPlot(CRX_data['Debt'])
plotDistPlot(CRX_data['YearsEmployed'])
plotDistPlot(CRX_data['CreditScore'])
plotDistPlot(CRX_data['Income'])


# In[80]:


# Applying logarithmic scale to correct skewness
num_cols = list(CRX_data.select_dtypes(exclude='object'))
CRX_data[num_cols] = CRX_data[num_cols].apply(lambda x: np.log(x + 1))


# In[81]:


# check for approved
sns.countplot(data = CRX_data, x = 'Approved')
plt.show()


# In[82]:


#converting non-numeric to numeric values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# # Looping for each object type column
#Using label encoder to convert into numeric types
for col in CRX_data:
    if CRX_data[col].dtypes=='object':
        CRX_data[col]=le.fit_transform(CRX_data[col])


# In[83]:


# Removing the feature which are not important and converting to NumPy array
CRX_data = CRX_data.drop(['DriversLicense', 'ZipCode'], axis=1)
CRX_data = CRX_data.values


# In[84]:


from sklearn.model_selection import train_test_split

# Creating new variable to input features and labels
X,y = CRX_data[:,0:13] , CRX_data[:,13]

# Spliting the data into training and testing sets
X_train, X_test, y_train, Y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=1)


# In[85]:


# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Scaling X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)


# In[86]:


# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Fitting logistic regression with default parameter values
logreg = LogisticRegression()
logreg.fit(rescaledX_train, y_train)


# In[87]:


# Import confusion_matrix
from sklearn.metrics import confusion_matrix

# Using the trained model to predict instances from the test set
y_pred = logreg.predict(rescaledX_test)

# Getting the accuracy score of predictive model
print("Logistic regression classifier has accuracy of: ", logreg.score(rescaledX_test, Y_test))

# Evaluate the confusion_matrix
confusion_matrix(Y_test, y_pred)


# In[88]:


# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define the grid of values for tol and max_iter
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]

# Create a dictionary
param_grid = dict(tol=tol, max_iter=max_iter)

# Initializing GridSearchCV
grid_model = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, cv=5)

# Rescaling the entire data set with all the predictive features
rescaledX = scaler.fit_transform(X)

# Calculating and summarizing the final results
grid_model_result = grid_model.fit(rescaledX, y)
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_ 
print("Best: %f using %s" %  (best_score, best_params))


# In[89]:


# Fitting logistic regression with best parameter values from gridsearch
logreg2 = LogisticRegression(max_iter=100, tol=0.01)
logreg2.fit(rescaledX_train, y_train)


# In[90]:


# Using the trained model to predict instances from the test set
y_pred2 = logreg2.predict(rescaledX_test)

# Getting the accuracy score of predictive model
print("Logistic regression classifier has accuracy of: ", logreg2.score(rescaledX_test, Y_test))

# Evaluate the confusion_matrix
confusion_matrix(Y_test, y_pred2)

