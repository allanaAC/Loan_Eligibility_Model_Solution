from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import logging

# Function to split data
def split_data(df):
    try:
        x = df.drop('Loan_Status', axis=1)
        y = df['Loan_Status']
        return train_test_split(x, y, test_size=0.2, random_state=123, stratify=y)
    
    except Exception as e:
        logging.error(" Error in split_data data: {}". format(e))
        
def scale_data(xtrain, xtest):
    try:
        scaler = MinMaxScaler()
        xtrain_scaled = scaler.fit_transform(xtrain)
        xtest_scaled = scaler.transform(xtest)
        return xtrain_scaled, xtest_scaled
    
    except Exception as e:
        logging.error(" Error in scale_data data: {}". format(e))

def train_logistic_regression(xtrain_scaled, ytrain):
    try:
        model = LogisticRegression()
        model.fit(xtrain_scaled, ytrain)
        return model
    
    except Exception as e:
        logging.error(" Error in train_logistic_regression data: {}". format(e))
        
def train_random_forest(xtrain, ytrain):
    try:
        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_features='sqrt')
        model.fit(xtrain, ytrain)
        return model
    
    except Exception as e:
        logging.error(" Error in train_random_forest data: {}". format(e))
