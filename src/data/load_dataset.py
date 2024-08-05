#this function is to load data 
import pandas as pd
import numpy as np
import logging

data_path = "src/data/credit.csv"
def load_data(data_path):
    
    try:
        return pd.read_csv(data_path)
    
    except Exception as e:
        logging.error(" Error in loading data: {}". format(e))

def preprocess_data(df):
    try:
        df = df.drop('Loan_ID', axis=1)
        df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents','Education','Self_Employed','Property_Area'], dtype='int')
        return df
    
    except Exception as e:
        logging.error(" Error in preprocess_data data: {}". format(e))