import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score

# Load data from CSV file
file_path = 'data/review_sentiments_stars.csv'  # Update with the path to your actual data file
df = pd.read_csv(file_path)

# Splitting the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Define features and target
features = ['price', 'service', 'ambiance', 'food']
target = 'stars'

# Prepare XGBoost DMatrices, handling NaN values natively
dtrain = xgb.DMatrix(train_df[features], label=train_df[target])
dtest = xgb.DMatrix(test_df[features], label=test_df[target])

# Specify model training parameters
params = {
    'max_depth': 3,  # Maximum depth of a tree
    'eta': 0.1,  # Learning rate
    'objective': 'reg:squarederror',  # Loss function for regression
    'eval_metric': 'mlogloss'
}
num_rounds = 100  # Number of boosting rounds

# Training the model
bst = xgb.train(params, dtrain, num_rounds)

def test():
    # Predicting the test set
    test_predictions = bst.predict(dtest)
    # Displaying the predictions
    print("Test Predictions:", test_predictions)
    actuals = test_df[target].values
    print("Actuals vs Predictions:")
    print(np.column_stack((actuals, test_predictions))) 

def eval():
    # Predicting the test set
    test_predictions = np.rint(bst.predict(dtest)).astype(int)

    # Displaying the predictions
    print("Test Predictions:", test_predictions)


    actuals = np.rint(test_df[target].values).astype(int)
    print("Actuals vs Predictions:")
    print(np.column_stack((actuals, test_predictions)))  # Adding 1 to match the original label scale

    # Evaluation
    cm = confusion_matrix(actuals, test_predictions)
    accuracy = accuracy_score(actuals, test_predictions)
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy)

test()
eval()