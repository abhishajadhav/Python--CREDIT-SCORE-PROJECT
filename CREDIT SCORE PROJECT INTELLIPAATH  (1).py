#!/usr/bin/env python
# coding: utf-8

# You are working as a data scientist in a 
# global finance company. Over the 
# years, the company has collected basic 
# bank details and gathered a lot of 
# credit-related information. The 
# management wants to build an 
# intelligent system to segregate the 
# people into credit score brackets to 
# reduce the manual efforts. Given a 
# person’s credit-related information, 
# build a machine learning model that 
# can classify the credit score

# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[34]:


df = pd.read_csv(r"C:\Users\DELL\Desktop\credit_score.csv")


# In[35]:


df


# In[36]:


df = df.drop(columns = ['ID', 'Customer_ID', 'Name', 'SSN','Credit_History_Age'])


# In[37]:


df.info()


# In[38]:


df = df.drop(columns = ['Type_of_Loan'])


# In[39]:


df


# In[40]:


df.describe()


# In[41]:


df.isnull().sum()


# In[42]:


df.tail()


# In[43]:


# Clean and convert 'Age' by removing non-numeric characters and converting to int
df['Age'] = df['Age'].astype(str).str.replace(r'[^0-9]', '', regex=True).astype(int)

# Clean and convert 'Annual_Income' by removing non-numeric characters and converting to float
df['Annual_Income'] = df['Annual_Income'].astype(str).str.replace(r'[^0-9.]', '', regex=True)
df['Annual_Income'] = pd.to_numeric(df['Annual_Income'], errors='coerce').fillna(0).astype(float)

# Clean and convert 'Num_of_Loan' by removing non-numeric characters and converting to int
df['Num_of_Loan'] = df['Num_of_Loan'].astype(str).str.replace(r'[^0-9]', '', regex=True)
df['Num_of_Loan'] = pd.to_numeric(df['Num_of_Loan'], errors='coerce').fillna(0).astype(int)

# Clean and convert 'Num_of_Delayed_Payment' by removing non-numeric characters and converting to int
df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].astype(str).str.replace(r'[^0-9]', '', regex=True)
df['Num_of_Delayed_Payment'] = pd.to_numeric(df['Num_of_Delayed_Payment'], errors='coerce').fillna(0).astype(int)

# Clean and convert 'Changed_Credit_Limit' by removing non-numeric characters and converting to int
df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].astype(str).str.replace(r'[^0-9]', '', regex=True)
df['Changed_Credit_Limit'] = pd.to_numeric(df['Changed_Credit_Limit'], errors='coerce').fillna(0).astype(int)

# Clean and convert 'Outstanding_Debt' by removing non-numeric characters and converting to int
df['Outstanding_Debt'] = df['Outstanding_Debt'].astype(str).str.replace(r'[^0-9]', '', regex=True)
df['Outstanding_Debt'] = pd.to_numeric(df['Outstanding_Debt'], errors='coerce').fillna(0).astype(int)

# Clean and convert 'Amount_invested_monthly' by removing non-numeric characters and converting to int
df['Amount_invested_monthly'] = df['Amount_invested_monthly'].astype(str).str.replace(r'[^0-9]', '', regex=True)
df['Amount_invested_monthly'] = pd.to_numeric(df['Amount_invested_monthly'], errors='coerce').fillna(0).astype(int)

# Clean and convert 'Monthly_Balance' by removing non-numeric characters and converting to int
df['Monthly_Balance'] = df['Monthly_Balance'].astype(str).str.replace(r'[^0-9]', '', regex=True)
df['Monthly_Balance'] = pd.to_numeric(df['Monthly_Balance'], errors='coerce').fillna(0).astype(int)

# Convert 'Payment_of_Min_Amount' (Yes/No) to 1/0
df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace(["Yes", "No"], [1, 0])

# Convert 'Credit_Score' (Poor, Standard, Good) to 0, 1, 2
df['Credit_Score'] = df['Credit_Score'].replace(["Poor", "Standard", "Good"], [0, 1, 2])

# Convert 'Payment_Behaviour' where needed, e.g., handling outlier cases
df['Payment_Behaviour'] = df['Payment_Behaviour'].replace("!@9#%8", np.nan)

print(df.dtypes)
print(df.head())


# In[44]:


df = df.fillna(method="ffill")
df = df.fillna(method="bfill")    #foward and back fill method for null values


# In[45]:


df.isnull().sum()


# In[46]:


df.info()


# In[47]:


#removing outliers from age since all other columns values are relevant 
sns.boxplot(df['Age'])                               
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[48]:


col_names = ['Age']
Q1 = df.Age.quantile(0.25)
Q3 = df.Age.quantile(0.75)
IQR = Q3 - Q1


# In[49]:


data= df[(df.Age >= Q1 - 1.5*IQR) & (df.Age <= Q3 + 1.5*IQR)]


# In[50]:


sns.boxplot(data['Age'])
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[53]:


#Performing One Hot Encoding for categorical features of a dataframe
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Month"]= le.fit_transform(df["Month"])
df["Occupation"]= le.fit_transform(df["Occupation"])
df["Payment_Behaviour"]= le.fit_transform(df["Payment_Behaviour"])


# In[54]:


df.info()


# In[55]:


df['Payment_of_Min_Amount'] = pd.to_numeric(df['Payment_of_Min_Amount'], errors='coerce').fillna(0).astype(int)
df['Credit_Mix'] = pd.to_numeric(df['Credit_Mix'], errors='coerce').fillna(0).astype(int)


# In[56]:


X = df.drop(columns=['Credit_Score'])
y = df["Credit_Score"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state=42)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[57]:


pd.DataFrame({"Actual_value": y_test, "Predicted_Value":y_pred})


# In[58]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
accuracy_score(y_test,y_pred)


# In[59]:


pd.DataFrame({"Actual_value": y_test, "Predicted_Value":y_pred})


# In[60]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
accuracy_score(y_test,y_pred)


# In[61]:


pd.DataFrame({"Actual_value": y_test, "Predicted_Value":y_pred})


# In[ ]:




