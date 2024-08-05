import warnings
from src.data.load_dataset import load_data, preprocess_data
from src.visualization.visualize import initial_data_analysis
from src.feature.build_features import impute_missing_values
from src.model.train_model import split_data, scale_data, train_logistic_regression, train_random_forest
from src.model.predict_model import evaluate_model, cross_validate_model


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # Load data
    df = load_data('src/data/credit.csv')

    # Initial data analysis
    initial_data_analysis(df)

    # Impute missing values
    df = impute_missing_values(df)

    # Preprocess data
    df = preprocess_data(df)

    # Split data
    xtrain, xtest, ytrain, ytest = split_data(df)

    # Scale data
    xtrain_scaled, xtest_scaled = scale_data(xtrain, xtest)
    
    # Train Logistic Regression model
    lr_model = train_logistic_regression(xtrain_scaled, ytrain)

    # Evaluate Logistic Regression model
    print("Logistic Regression Evaluation:")
    evaluate_model(lr_model, xtest_scaled, ytest)

    # Train Random Forest model
    rf_model = train_random_forest(xtrain, ytrain)

    # Evaluate Random Forest model
    print("Random Forest Evaluation:")
    evaluate_model(rf_model, xtest, ytest)

    # Cross-validate Logistic Regression model
    print("Logistic Regression Cross-Validation:")
    cross_validate_model(lr_model, xtrain_scaled, ytrain)

    # Cross-validate Random Forest model
    print("Random Forest Cross-Validation:")
    cross_validate_model(rf_model, xtrain_scaled, ytrain)
    
    
    