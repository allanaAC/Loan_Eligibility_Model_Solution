from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import logging

# # Function to predict and evaluate model 
def evaluate_model(model, xtest_scaled, ytest):
    try:
        ypred = model.predict(xtest_scaled)
        accuracy = accuracy_score(ytest, ypred)
        cm = confusion_matrix(ytest, ypred)
        print(f'Accuracy: {accuracy}\nConfusion Matrix:\n{cm}')
        return accuracy, cm

    except Exception as e:
        logging.error(" Error in evaluate_model data: {}". format(e))
        
def cross_validate_model(model, xtrain_scaled, ytrain):
    try:
        kfold = KFold(n_splits=5)
        scores = cross_val_score(model, xtrain_scaled, ytrain, cv=kfold)
        print(f'Accuracy scores: {scores}')
        print(f'Mean accuracy: {scores.mean()}')
        print(f'Standard deviation: {scores.std()}')
        
    except Exception as e:
        logging.error(" Error in cross_validate_model data: {}". format(e))