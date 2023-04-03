import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read the data
data = pd.read_csv("../data/New_Train_Data.csv")

# Preprocess the data
X = data.iloc[:, :-1].values
y = data["cost rank"].values - 1  # transfer the label to 0-3

# Scale or normalize the numerical range of input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.0001, 0.001, 0.01],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Initialize the GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=200, 
                                 learning_rate=0.001, 
                                 max_depth=5, 
                                 min_samples_split=4, 
                                 min_samples_leaf=2, 
                                 max_features='sqrt', 
                                 random_state=42)

# Initialize GridSearchCV with the model and parameter grid
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, verbose=0, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X, y)

# Find and print the best parameters
print("Best parameters found: ", grid_search.best_params_)


# K-Fold cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_accuracies = []
fold_precisions = []
fold_recalls = []
fold_f1_scores = []

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train and evaluate the model with the best parameters
    best_clf = grid_search.best_estimator_
    #best_clf = clf
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    fold_accuracies.append(accuracy)
    fold_precisions.append(precision)
    fold_recalls.append(recall)
    fold_f1_scores.append(f1)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    
avg_accuracy = np.mean(fold_accuracies)
avg_precision = np.mean(fold_precisions)
avg_recall = np.mean(fold_recalls)
avg_f1_score = np.mean(fold_f1_scores)

print(f"K-Fold Validation Results: Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1_score:.4f}")