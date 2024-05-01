#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the training and testing datasets
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

# Step 2: Select relevant columns
train_features = train_dataset[['GrLivArea', 'BedroomAbvGr', 'HalfBath', 'FullBath']]

test_features = test_dataset[['GrLivArea', 'BedroomAbvGr', 'HalfBath', 'FullBath']]

# Step 3: Train the model
model = LinearRegression()
model.fit(train_features, train_target)

# Step 4: Evaluate the model
test_pred = model.predict(test_features)
mse = mean_squared_error(test_target, test_pred)
print("Mean Squared Error:", mse)

# Step 5: Save the predicted values to a new CSV file
predictions_df = pd.DataFrame({'Actual': test_target, 'Predicted': test_pred})
predictions_df.to_csv('predicted_values.csv', index=False)


# In[28]:


import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the training and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Extract features and target variable from training dataset
X_train = train_data[['GrLivArea', 'BedroomAbvGr', 'HalfBath', 'FullBath']]
y_train = train_data['SalePrice']

# Extract features from test dataset
X_test = test_data[['GrLivArea', 'BedroomAbvGr', 'HalfBath', 'FullBath']]

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict house prices on test data
y_pred = model.predict(X_test)

# Round predicted prices to two decimal places
y_pred_rounded = [round(price, 2) for price in y_pred]

# Add rounded predicted prices as a new column to the test dataset
test_data['PredictedPrice'] = y_pred_rounded

# Save the test dataset with rounded predicted prices and selected columns to a new CSV file
test_data.to_csv('predicted_prices.csv', columns=['Id','PredictedPrice'], index=False)


# In[13]:


model.coef_


# In[14]:


model.intercept_


# In[15]:


896*108.22377873+2*-27911.62493864+1*30380.78357956+47997.69971509665


# In[ ]:




