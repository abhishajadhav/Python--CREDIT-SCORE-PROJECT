# Python--CREDIT-SCORE-PROJECT

This project aims to build an intelligent system to classify individuals into credit score brackets using machine learning. The system analyzes customer credit-related information to automate the segregation of people into credit score categories.

Dataset Attributes
The dataset includes the following attributes:
ID: Unique identifier for each entry
Customer_ID: Unique identifier for a person
Age, Occupation, Annual_Income, Monthly_Inhand_Salary: Personal and financial details
Num_Bank_Accounts, Num_Credit_Card, Num_of_Loan: Banking and credit card information
Interest_Rate, Delay_from_due_date, Credit_Utilization_Ratio: Credit details
Outstanding_Debt, Credit_Mix, Payment_Behaviour: Credit repayment behavior
Credit_Score: The target variable representing the credit score bracket

Data Cleaning:
Removed special characters, handled missing values, and converted categorical data into numeric values.
Fixed data types (int, float) for relevant columns.

Outlier Removal:
Removed outliers in the Age column to ensure data quality.

Feature Selection:
Selected relevant features based on their importance for the classification task.
Machine Learning Models:

Logistic Regression: For binary classification based on credit score brackets.
Decision Tree: To model credit score classification with tree-based learning.
Random Forest: Used for ensemble learning, achieving the best results.

Key Results
Random Forest classifier achieved high accuracy in predicting credit score brackets.

Technologies Used
Python Libraries: Pandas, NumPy, Scikit-learn, Matplotlib
