import pandas as pd
import logging
   
def impute_missing_values(df):
    try:
        df['Gender'].fillna('Male', inplace=True)
        df['Married'].fillna(df['Married'].mode()[0], inplace=True)
        df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
        df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
        df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
        df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
        df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
        #print(df.isnull().sum())
        return df
    
    except Exception as e:
        logging.error(" Error in impute_missing_values data: {}". format(e))
        
        
