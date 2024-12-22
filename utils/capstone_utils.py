import pandas as pd 
import numpy as np 

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate a fitted model on training and test data with various metrics.
    
    Parameters:
    - model: The fitted model to evaluate.
    - X_train: Features for the training data.
    - y_train: Target for the training data.
    - X_test: Features for the test data.
    - y_test: Target for the test data.
    
    Outputs:
    - Training accuracy.
    - Test accuracy.
    - A confusion matrix display.
    - Confusion matrix in DataFrame format.
    - Classification report.
    - ROC curve.
    - Test AUC score.
    - Train AUC score.
    """
    # Training and test accuracy
    print(f'Training Score: {model.score(X_train, y_train)}')
    print(f'Test Score: {model.score(X_test, y_test)}')
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Confusion Matrix Display
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.show()  # Display the plot immediately
    
    # Confusion Matrix DataFrame
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred),
                                       columns=['Predicted Negative', 'Predicted Positive'], 
                                       index=['True Negative', 'True Positive'])
    print("Confusion Matrix (DataFrame):")
    display(confusion_matrix_df)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # ROC Curve and AUC
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_proba_train = model.predict_proba(X_train)[:, 1]
    fprs, tprs, _ = roc_curve(y_test, y_proba_test)
    roc_auc = roc_auc_score(y_test, y_proba_test)
    fprs_train, tprs_train, _ = roc_curve(y_train, y_proba_train)
    roc_auc_train = roc_auc_score(y_train, y_proba_train)
    
    plt.figure()
    plt.plot(fprs_train, tprs_train, color='darkorange', lw=2, label=f'Train ROC curve (area = {roc_auc_train:.2f})')
    plt.plot(fprs, tprs, lw=2, label=f'Test ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC and AUC')
    plt.legend(loc="best")
    plt.show()
    
    print(f"Test AUC score: {roc_auc}")
    print(f"Train AUC score: {roc_auc_train}")


def find_correlated_pairs(corr_matrix, threshold):
    """
    Find pairs of features in the correlation matrix where the absolute value
    of the correlation coefficient is greater than a specified threshold.
    
    Parameters:
    - corr_matrix: A pandas DataFrame representing the correlation matrix.
    - threshold: A float representing the threshold for the absolute value of the correlation coefficient.
    
    Returns:
    - A list of tuples, where each tuple contains a pair of feature names and their correlation coefficient,
      with the absolute value of the coefficient greater than the threshold. Each pair is returned once,
      ignoring self-correlations.
    """
    correlated_pairs = []
    
    # Iterate over the upper triangle of the correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            # Check if the absolute correlation coefficient exceeds the threshold
            if abs(corr_matrix.iloc[i, j]) > threshold:
                # Append the pair and the correlation coefficient to the list
                correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                
    return correlated_pairs

def get_model_coefficients(training_set, model):
    """
    Generate a DataFrame with feature names, their corresponding coefficients from a logistic regression model,
    the absolute value of these coefficients, and the odds ratios.
    
    Parameters:
    - training_set: A DataFrame containing features.
    - model: A fitted logistic regression model object from scikit-learn.
    
    Returns:
    - DataFrame: A DataFrame sorted by the absolute values of the coefficients in descending order.
                 The DataFrame contains columns for feature names, coefficients, absolute coefficients, and odds ratios.
    """
    
    # Create a DataFrame with feature names and their coefficients
    coef_df = pd.DataFrame({
        'Feature': training_set.columns,
        'Coefficients': model.coef_[0]
    })
    
    # Calculate the absolute value of coefficients
    coef_df['Absolute Coef'] = np.abs(coef_df['Coefficients'])
    
    # Calculate the odds ratios
    coef_df['Odds_Ratio'] = np.exp(coef_df['Coefficients'])
    
    # Sort by absolute value of coefficients, descending
    coef_df_sorted = coef_df.sort_values('Absolute Coef', ascending=False).reset_index(drop=True)
    
    return coef_df_sorted.T

def summarize_lr_models(model_names, models, train_sets, test_sets, y_train_list, y_test_list):
    """
    Summarizes and compares pre-fitted logistic regression models.
    
    Parameters:
    - model_names: A list of names (strings) assigned to each model.
    - models: A list of pre-fitted logistic regression model objects.
    - train_sets: A list of training sets corresponding to each model.
    - test_sets: A list of test sets corresponding to each model.
    - y_train_list: A list of training labels.
    - y_test_list: A list of test labels.
    
    Returns:
    - DataFrame: A pandas DataFrame containing the summary of each model.
    """
    # Import metrics libraries
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Initialize a list to store each model's summary
    summaries = []
    
    # Iterate over each model and its corresponding datasets
    for model_name, model, X_train, X_test, y_train, y_test in zip(model_names, models, train_sets, test_sets, y_train_list, y_test_list):
        # Compute the number of features
        num_features = X_train.shape[1]
        
        # List containing all the feature names
        feature_names = list(X_train.columns)
        
        # Make predictions on both the training and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate accuracy, precision, recall, and f1 score
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        
        # Calculate AUC score
        # Note: roc_auc_score expects probability scores for the positive class
        y_test_prob = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_test_prob)
        
        # Append the summary for the current model to the summaries list
        summaries.append([model_name, num_features, feature_names, train_accuracy, test_accuracy, precision, recall, f1, auc_score])
    
    # Convert the summaries list into a DataFrame
    summary_df = pd.DataFrame(summaries, columns=['Model Name', 'Number of Features', 'Feature Names', 'Training Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC Score'])
    
    return summary_df