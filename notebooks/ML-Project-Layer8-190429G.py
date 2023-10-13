# %% [markdown]
# Import required libraries and modules

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sklearn.model_selection import RandomizedSearchCV

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score


# %% [markdown]
# Import training, validation and testing datasets

# %%
# file paths for the datasets
train_path = './../data/layer 8/train.csv'
valid_path = './../data/layer 8/valid.csv'
test_path = './../data/layer 8/test.csv'

# Load the train dataset
train_data = pd.read_csv(train_path)

# Load the valid dataset
valid_data = pd.read_csv(valid_path)

# Load the test dataset
test_data = pd.read_csv(test_path)

# %% [markdown]
# Get List of IDs in test dataset

# %%
ID = test_data['ID'].to_numpy()

# %% [markdown]
# Drop the ID column

# %%
test_data = test_data.drop(columns=['ID'])

# %% [markdown]
# Construct validation and training data for each label

# %%
# Datasets for each label in train dataset
train_data_label1 = train_data.drop(columns=['label_2', 'label_3', 'label_4'])
train_data_label2 = train_data.drop(columns=['label_1', 'label_3', 'label_4'])
train_data_label3 = train_data.drop(columns=['label_1', 'label_2', 'label_4'])
train_data_label4 = train_data.drop(columns=['label_1', 'label_2', 'label_3'])

# Datasets for each label in train dataset
valid_data_label1 = valid_data.drop(columns=['label_2', 'label_3', 'label_4'])
valid_data_label2 = valid_data.drop(columns=['label_1', 'label_3', 'label_4'])
valid_data_label3 = valid_data.drop(columns=['label_1', 'label_2', 'label_4'])
valid_data_label4 = valid_data.drop(columns=['label_1', 'label_2', 'label_3'])

# %% [markdown]
# ### Check for outliers

# %%
train_features = train_data.drop(columns=['label_1', 'label_2', 'label_3', 'label_4'])
valid_features = valid_data.drop(columns=['label_1', 'label_2', 'label_3', 'label_4'])

# %%
train_features_array = train_features.to_numpy()

valid_features_array = valid_features.to_numpy()

test_features_array = test_data.to_numpy()

# %%
train_features.describe()

# %% [markdown]
# Outliers are present

# %% [markdown]
# Isolation Forest

# %%
# identify outliers in the training dataset
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(train_features_array)

# %% [markdown]
# Minimum Covariance Determinant

# %%
# identify outliers in the training dataset
ee = EllipticEnvelope(contamination=0.01)
yhat = ee.fit_predict(train_features_array)

# %% [markdown]
# Local Outlier Factor

# %%
# identify outliers in the training dataset
lof = LocalOutlierFactor()
yhat = lof.fit_predict(train_features_array)

# %% [markdown]
# One-Class SVM

# %%
# identify outliers in the training dataset
ee = OneClassSVM(nu=0.01)
yhat = ee.fit_predict(train_features_array)

# %% [markdown]
# Remove outliers

# %%
# select all rows that are not outliers
mask = yhat != -1

train_features_array, train_labels_array = train_features_array[mask, :], train_labels_array[mask, :]

# %%
train_features_array.shape

# %% [markdown]
# # Label 1

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# Predict label 1 with feature engineering steps and techniques.  
# Use feature selection based on correlation matrix and feature extraction based on PCA

# %% [markdown]
# ### Data cleaning

# %% [markdown]
# Process the data to remove null values for labels and estimate missing values in features

# %% [markdown]
# Drop the columns where there are null values for the lables in the training dataset

# %%
# Check for null values in train dataset
print(f"train_data_label1 shape before : {train_data_label1.shape}")
train_null_counts = train_data_label1.isnull().sum()
print(f"train null counts before : \n{train_null_counts}")

# Drop rows with null values in the target labels for train dataset
train_data_label1 = train_data_label1.dropna(subset=train_data_label1.columns[-1:], how='any')
print(f"train_data_label1 shape after : {train_data_label1.shape}")

# %% [markdown]
# Fill the null values in the features with their means in the train, valid and test datasets.

# %%
# Fill null values with mean in train dataset
train_data_label1 = train_data_label1.fillna(train_data_label1.mean())

# Fill null values with mean in valid dataset
valid_data_label1 = valid_data_label1.fillna(valid_data_label1.mean())

# Fill null values with mean in test dataset
test_data = test_data.fillna(test_data.mean())

# %% [markdown]
# Extract the features and the label

# %%
train_features_label1 = train_data_label1.iloc[:, :-1]
train_label1 = train_data_label1.iloc[:, -1]

valid_features_label1 = valid_data_label1.iloc[:, :-1]
valid_label1 = valid_data_label1.iloc[:, -1]

test_features_label1 = test_data

# %% [markdown]
# Visualize the distribution of the training label 1

# %%
# Plotting the distribution of train_label1
labels, counts = np.unique(train_label1, return_counts=True)

plt.figure(figsize=(22, 6))
plt.xticks(labels)
plt.bar(labels, counts)
plt.xlabel('Target Label 1')
plt.ylabel('Frequency')
plt.title('Distribution of Target Label 1')
plt.show()

# %% [markdown]
# ### Feature Standardization

# %% [markdown]
# Standardize the features of all datasets using Robust scaler beacause outliers are present

# %%
# Standardize the features
scaler = RobustScaler()
standardized_train_features_label1 = scaler.fit_transform(train_features_label1)
standardized_valid_features_label1 = scaler.transform(valid_features_label1)
standardized_test_features_label1 = scaler.transform(test_features_label1)

# %% [markdown]
# ### Feature Extraction

# %% [markdown]
# Extract can combine the features that are highly significant in predicting the label using Principal Componenet Analysis(PCA)  
# Extract the features that can explain the variance of the label to 97%  
# Display the resulting explained variances of each principal component  

# %%
variance_threshold = 0.97

# Apply PCA with the determined number of components
pca = PCA(n_components=variance_threshold, svd_solver='full')

pca_train_features_label1 = pca.fit_transform(standardized_train_features_label1)
pca_valid_features_label1 = pca.transform(standardized_valid_features_label1)
pca_test_features_label1 = pca.transform(standardized_test_features_label1)

# Explained variance ratio after dimensionality reduction
explained_variance_ratio_reduced = pca.explained_variance_ratio_

# Plot explained variance ratio
plt.figure(figsize=(18, 10))
plt.bar(range(1, pca_train_features_label1.shape[1] + 1), explained_variance_ratio_reduced)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio per Principal Component (Reduced)')
plt.show()

# Display the reduced train feature matrix
print("Reduced Train feature matrix shape: {}".format(pca_train_features_label1.shape))
# Display the reduced valid feature matrix
print("Reduced valid feature matrix shape: {}".format(pca_valid_features_label1.shape))
# Display the reduced test feature matrix
print("Reduced test feature matrix shape: {}".format(pca_test_features_label1.shape))

# %% [markdown]
# ### Feature Selection

# %% [markdown]
# Calculate the correlation matrix of the training data features

# %%
#Calculate the correlation matrix
correlation_matrix = pd.DataFrame(pca_train_features_label1).corr()

mask = np.triu(np.ones_like(correlation_matrix))

# Create a heatmap of the correlation matrix using seaborn
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, mask=mask)
plt.title("Correlation Matrix")
plt.show()

# %% [markdown]
# Identify the features that are highly correlated with each other using the training dataset

# %%
# Set the threshold for correlation
correlation_threshold = 0.9

highly_correlated = set()

# Find highly correlated features
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated.add(colname)

print(highly_correlated)

# %% [markdown]
# Remove the previously identified highly correlated features from all the datasets

# %%
# Remove highly correlated features
pca_train_features_label1 = pd.DataFrame(pca_train_features_label1).drop(columns=highly_correlated)
pca_valid_features_label1 = pd.DataFrame(pca_valid_features_label1).drop(columns=highly_correlated)
pca_test_features_label1 = pd.DataFrame(pca_test_features_label1).drop(columns=highly_correlated)

# %% [markdown]
# Display the resulting feature shapes of the datasets

# %%
# Display the filtered train feature count
print("Filtered train features: {}".format(pca_train_features_label1.shape))

# Display the filtered valid feature count
print("Filtered valid features: {}".format(pca_valid_features_label1.shape))

# Display the filtered test feature count
print("Filtered test features: {}".format(pca_test_features_label1.shape))

# %% [markdown]
# Convert dataframe to numpy array

# %%
pca_train_features_label1 = pca_train_features_label1.to_numpy()
train_label1 = train_label1.to_numpy()

pca_valid_features_label1 = pca_valid_features_label1.to_numpy()
valid_label1 = valid_label1.to_numpy()

pca_test_features_label1 = pca_test_features_label1.to_numpy()

# %% [markdown]
# ## Hyperparameter tuning

# %% [markdown]
# Define parameters for random search

# %%
knn_grid_params = {
    'n_neighbors' : [3, 5, 7, 9, 11, 13],
    'weights' : ['uniform', 'distance'],
    'metric' : ['minkowski', 'euclidean', 'manhattan', 'hamming']
}

rf_grid_params = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
}

svm_grid_params = {
    'C': [0.1, 1, 10, 30, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'auto', 'scale'],
    'kernel': ['rbf', 'linear']
}

# %% [markdown]
# Execute random search for KNN, Random forest and SVM

# %%
classification_models_params = [
    ('K Neighbors', KNeighborsClassifier(), knn_grid_params),
    ('Random Forest', RandomForestClassifier(), rf_grid_params),
    ('SVM', SVC(), svm_grid_params)
]

for model_name, model, grid_params in classification_models_params:
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rs = RandomizedSearchCV(
        estimator = model,
        param_distributions = grid_params,
        n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1
    )
    rs_result = rs.fit(pca_train_features_label1, train_label1)

    print(f"best score for {model_name} : {rs_result.best_score_}")
    print(f"best hyper parameters for {model_name} : {rs_result.best_params_}")

# %% [markdown]
# ## Model Selection

# %% [markdown]
# Select the model that best predicts the valid and test datasets based on accuracy, precision and recall

# %%
# Define a list of classification models
classification_models = [
    # ('K Neighbors', KNeighborsClassifier(n_neighbors=5, metric='minkowski', weights='distance')),
    # ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC(C=100, gamma='scale', kernel='rbf')),
]

# The best model is SVM then KNN then Random Forest

# Number of features used in PCA
num_features = pca_train_features_label1.shape[1]
print(f"Number of features: {num_features}\n")

# Train and evaluate each classification model
for model_name, model in classification_models:
    # Train the model on the training data
    model.fit(pca_train_features_label1, train_label1)

    # Predict on the train data
    y_pred_train_label1 = model.predict(pca_train_features_label1)

    # Calculate metrics for classification evaluation
    accuracy = accuracy_score(train_label1, y_pred_train_label1)
    precision = precision_score(train_label1, y_pred_train_label1, average='macro', zero_division=1)
    recall = recall_score(train_label1, y_pred_train_label1, average='macro')

    print(f"Metrics for {model_name} on train data:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print("\n")

    # Predict on the validation data
    y_pred_valid_label1 = model.predict(pca_valid_features_label1)

    # Calculate metrics for classification evaluation on validation data
    accuracy = accuracy_score(valid_label1, y_pred_valid_label1)
    precision = precision_score(valid_label1, y_pred_valid_label1, average='macro', zero_division=1)
    recall = recall_score(valid_label1, y_pred_valid_label1, average='macro')

    print(f"Metrics for {model_name} on validation data:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print("\n")

    # Predict on the test data
    y_pred_test_label1 = model.predict(pca_test_features_label1)


# %% [markdown]
# # Label 2

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# Predict label 2 with feature engineering steps and techniques.  
# Use feature selection based on correlation matrix and feature extraction based on PCA

# %% [markdown]
# ### Data cleaning

# %% [markdown]
# Process the data to remove null values for labels and estimate missing values in features

# %% [markdown]
# Drop the columns where there are null values for the lables in the training dataset

# %%
# Check for null values in train dataset
print(f"train_data_label2 shape before : {train_data_label2.shape}")
train_null_counts = train_data_label2.isnull().sum()
print(f"train null counts before : \n{train_null_counts}")

# Drop rows with null values in the target labels for train dataset
train_data_label2 = train_data_label2.dropna(subset=train_data_label2.columns[-1:], how='any')
print(f"train_data_label2 shape after : {train_data_label2.shape}")

# %% [markdown]
# Fill the null values in the features with their means in the train, valid and test datasets.

# %%
# Fill null values with mean in train dataset
train_data_label2 = train_data_label2.fillna(train_data_label2.mean())

# Fill null values with mean in valid dataset
valid_data_label2 = valid_data_label2.fillna(valid_data_label2.mean())

# Fill null values with mean in test dataset
test_data = test_data.fillna(test_data.mean())

# %% [markdown]
# Extract the features and the label

# %%
train_features_label2 = train_data_label2.iloc[:, :-1]
train_label2 = train_data_label2.iloc[:, -1].astype('int64')

valid_features_label2 = valid_data_label2.iloc[:, :-1]
valid_label2 = valid_data_label2.iloc[:, -1].astype('int64')

test_features_label2 = test_data

# %% [markdown]
# Visualize the distribution of the training label 2

# %%
# Plotting the distribution of train_label2
labels, counts = np.unique(train_label2, return_counts=True)

plt.figure(figsize=(22, 6))
plt.xticks(labels)
plt.bar(labels, counts)
plt.xlabel('Target Label 2')
plt.ylabel('Frequency')
plt.title('Distribution of Target Label 2')
plt.show()

# %% [markdown]
# ### Feature Standardization

# %% [markdown]
# Standardize the features of all datasets using Robust scaler beacause outliers are present

# %%
# Standardize the features
scaler = RobustScaler()
standardized_train_features_label2 = scaler.fit_transform(train_features_label2)
standardized_valid_features_label2 = scaler.transform(valid_features_label2)
standardized_test_features_label2 = scaler.transform(test_features_label2)

# %% [markdown]
# ### Feature Extraction

# %% [markdown]
# Extract can combine the features that are highly significant in predicting the label using Principal Componenet Analysis(PCA)  
# Extract the features that can explain the variance of the label to 97%  
# Display the resulting explained variances of each principal component  

# %%
variance_threshold = 0.97

# Apply PCA with the determined number of components
pca = PCA(n_components=variance_threshold, svd_solver='full')

pca_train_features_label2 = pca.fit_transform(standardized_train_features_label2)
pca_valid_features_label2 = pca.transform(standardized_valid_features_label2)
pca_test_features_label2 = pca.transform(standardized_test_features_label2)

# Explained variance ratio after dimensionality reduction
explained_variance_ratio_reduced = pca.explained_variance_ratio_

# Plot explained variance ratio
plt.figure(figsize=(18, 10))
plt.bar(range(1, pca_train_features_label2.shape[1] + 1), explained_variance_ratio_reduced)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio per Principal Component (Reduced)')
plt.show()

# Display the reduced train feature matrix
print("Reduced Train feature matrix shape: {}".format(pca_train_features_label2.shape))
# Display the reduced valid feature matrix
print("Reduced valid feature matrix shape: {}".format(pca_valid_features_label2.shape))
# Display the reduced test feature matrix
print("Reduced test feature matrix shape: {}".format(pca_test_features_label2.shape))

# %% [markdown]
# ### Feature Selection

# %% [markdown]
# Calculate the correlation matrix of the training data features

# %%
#Calculate the correlation matrix
correlation_matrix = pd.DataFrame(pca_train_features_label2).corr()

mask = np.triu(np.ones_like(correlation_matrix))

# Create a heatmap of the correlation matrix using seaborn
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, mask=mask)
plt.title("Correlation Matrix")
plt.show()

# %% [markdown]
# Identify the features that are highly correlated with each other using the traning dataset

# %%
# Set the threshold for correlation
correlation_threshold = 0.9

highly_correlated = set()

# Find highly correlated features
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated.add(colname)

print(highly_correlated)

# %% [markdown]
# Remove the previously identified highly correlated features from all the datasets

# %%
# Remove highly correlated features
pca_train_features_label2 = pd.DataFrame(pca_train_features_label2).drop(columns=highly_correlated)
pca_valid_features_label2 = pd.DataFrame(pca_valid_features_label2).drop(columns=highly_correlated)
pca_test_features_label2 = pd.DataFrame(pca_test_features_label2).drop(columns=highly_correlated)

# %% [markdown]
# Display the resulting feature shapes of the datasets

# %%
# Display the filtered train feature count
print("Filtered train features: {}".format(pca_train_features_label2.shape))

# Display the filtered valid feature count
print("Filtered valid features: {}".format(pca_valid_features_label2.shape))

# Display the filtered test feature count
print("Filtered test features: {}".format(pca_test_features_label2.shape))

# %% [markdown]
# Convert dataframe to numpy array

# %%
pca_train_features_label2 = pca_train_features_label2.to_numpy()
train_label2 = train_label2.to_numpy()

pca_valid_features_label2 = pca_valid_features_label2.to_numpy()
valid_label2 = valid_label2.to_numpy()

pca_test_features_label2 = pca_test_features_label2.to_numpy()

# %% [markdown]
# ## Hyperparameter tuning

# %% [markdown]
# Define parameters for random search

# %%
knn_grid_params = {
    'n_neighbors' : [3, 5, 7, 9, 11, 13],
    'weights' : ['uniform', 'distance'],
    'metric' : ['minkowski', 'euclidean', 'manhattan', 'hamming']
}

rf_grid_params = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
}

svm_grid_params = {
    'C': [0.1, 1, 10, 30, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'auto', 'scale'],
    'kernel': ['rbf', 'linear']
}

# %% [markdown]
# Execute random search for KNN, Random forest and SVM

# %%
classification_models_params = [
    ('K Neighbors', KNeighborsClassifier(), knn_grid_params),
    ('Random Forest', RandomForestClassifier(), rf_grid_params)
    ('SVM', SVC(), svm_grid_params)
]

for model_name, model, grid_params in classification_models_params:
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rs = RandomizedSearchCV(
        estimator = model,
        param_distributions = grid_params,
        n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1
    )
    rs_result = rs.fit(pca_train_features_label2, train_label2)

    print(f"best score for {model_name} : {rs_result.best_score_}")
    print(f"best hyper parameters for {model_name} : {rs_result.best_params_}")

# %% [markdown]
# ## Model Selection

# %% [markdown]
# Select the model that best predicts the valid and test datasets based on accuracy, precision and recall

# %%
# Define a list of classification models
classification_models = [
    # ('K Neighbors', KNeighborsClassifier(n_neighbors=3, metric='minkowski', weights='distance')),
    # ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC(C=100, gamma='scale', kernel='rbf', class_weight='balanced')),
]

# The best model is SVM then KNN then Random Forest

# Number of features used in PCA
num_features = pca_train_features_label2.shape[1]
print(f"Number of features: {num_features}\n")

# Train and evaluate each classification model
for model_name, model in classification_models:
    # Train the model on the training data
    model.fit(pca_train_features_label2, train_label2)

    # Predict on the train data
    y_pred_train_label2 = model.predict(pca_train_features_label2)

    # Calculate metrics for classification evaluation
    accuracy = accuracy_score(train_label2, y_pred_train_label2)
    precision = precision_score(train_label2, y_pred_train_label2, average='weighted', zero_division=1)
    recall = recall_score(train_label2, y_pred_train_label2, average='weighted')

    print(f"Metrics for {model_name} on train data:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print("\n")

    # Predict on the validation data
    y_pred_valid_label2 = model.predict(pca_valid_features_label2)

    # Calculate metrics for classification evaluation on validation data
    accuracy = accuracy_score(valid_label2, y_pred_valid_label2)
    precision = precision_score(valid_label2, y_pred_valid_label2, average='weighted', zero_division=1)
    recall = recall_score(valid_label2, y_pred_valid_label2, average='weighted')

    print(f"Metrics for {model_name} on validation data:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print("\n")

    # Predict on the test data
    y_pred_test_label2 = model.predict(pca_test_features_label2)


# %% [markdown]
# # Label 3

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# Predict label 3 with feature engineering steps and techniques.  
# Use feature selection based on correlation matrix and feature extraction based on PCA

# %% [markdown]
# ### Data cleaning

# %% [markdown]
# Process the data to remove null values for labels and estimate missing values in features

# %% [markdown]
# Drop the columns where there are null values for the lables in the training dataset

# %%
# Check for null values in train dataset
print(f"train_data_label3 shape before : {train_data_label3.shape}")
train_null_counts = train_data_label3.isnull().sum()
print(f"train null counts before : \n{train_null_counts}")

# Drop rows with null values in the target labels for train dataset
train_data_label3 = train_data_label3.dropna(subset=train_data_label3.columns[-1:], how='any')
print(f"train_data_label3 shape after : {train_data_label3.shape}")

# %% [markdown]
# Fill the null values in the features with their means in the train, valid and test datasets.

# %%
# Fill null values with mean in train dataset
train_data_label3 = train_data_label3.fillna(train_data_label3.mean())

# Fill null values with mean in valid dataset
valid_data_label3 = valid_data_label3.fillna(valid_data_label3.mean())

# Fill null values with mean in test dataset
test_data = test_data.fillna(test_data.mean())

# %% [markdown]
# Extract the features and the label

# %%
train_features_label3 = train_data_label3.iloc[:, :-1]
train_label3 = train_data_label3.iloc[:, -1]

valid_features_label3 = valid_data_label3.iloc[:, :-1]
valid_label3 = valid_data_label3.iloc[:, -1]

test_features_label3 = test_data

# %% [markdown]
# Visualize the distribution of the training label 3

# %%
# Plotting the distribution of train_label3
labels, counts = np.unique(train_label3, return_counts=True)

plt.figure(figsize=(22, 6))
plt.xticks(labels)
plt.bar(labels, counts)
plt.xlabel('Target Label 3')
plt.ylabel('Frequency')
plt.title('Distribution of Target Label 3')
plt.show()

# %% [markdown]
# ### Feature Standardization

# %% [markdown]
# Standardize the features of all datasets using Robust scaler beacause outliers are present

# %%
# Standardize the features
scaler = RobustScaler()
standardized_train_features_label3 = scaler.fit_transform(train_features_label3)
standardized_valid_features_label3 = scaler.transform(valid_features_label3)
standardized_test_features_label3 = scaler.transform(test_features_label3)

# %% [markdown]
# ### Feature Extraction

# %% [markdown]
# Extract can combine the features that are highly significant in predicting the label using Principal Componenet Analysis(PCA)  
# Extract the features that can explain the variance of the label to 95%  
# Display the resulting explained variances of each principal component  

# %%
variance_threshold = 0.95

# Apply PCA with the determined number of components
pca = PCA(n_components=variance_threshold, svd_solver='full')

pca_train_features_label3 = pca.fit_transform(standardized_train_features_label3)
pca_valid_features_label3 = pca.transform(standardized_valid_features_label3)
pca_test_features_label3 = pca.transform(standardized_test_features_label3)

# Explained variance ratio after dimensionality reduction
explained_variance_ratio_reduced = pca.explained_variance_ratio_

# Plot explained variance ratio
plt.figure(figsize=(18, 10))
plt.bar(range(1, pca_train_features_label3.shape[1] + 1), explained_variance_ratio_reduced)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio per Principal Component (Reduced)')
plt.show()

# Display the reduced train feature matrix
print("Reduced Train feature matrix shape: {}".format(pca_train_features_label3.shape))
# Display the reduced valid feature matrix
print("Reduced valid feature matrix shape: {}".format(pca_valid_features_label3.shape))
# Display the reduced test feature matrix
print("Reduced test feature matrix shape: {}".format(pca_test_features_label3.shape))

# %% [markdown]
# ### Feature Selection

# %% [markdown]
# Calculate the correlation matrix of the training data features

# %%
#Calculate the correlation matrix
correlation_matrix = pd.DataFrame(pca_train_features_label3).corr()

mask = np.triu(np.ones_like(correlation_matrix))

# Create a heatmap of the correlation matrix using seaborn
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, mask=mask)
plt.title("Correlation Matrix")
plt.show()

# %% [markdown]
# Identify the features that are highly correlated with each other using the traning dataset

# %%
# Set the threshold for correlation
correlation_threshold = 0.9

highly_correlated = set()

# Find highly correlated features
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated.add(colname)

print(highly_correlated)

# %% [markdown]
# Remove the previously identified highly correlated features from all the datasets

# %%
# Remove highly correlated features
pca_train_features_label3 = pd.DataFrame(pca_train_features_label3).drop(columns=highly_correlated)
pca_valid_features_label3 = pd.DataFrame(pca_valid_features_label3).drop(columns=highly_correlated)
pca_test_features_label3 = pd.DataFrame(pca_test_features_label3).drop(columns=highly_correlated)

# %% [markdown]
# Display the resulting feature shapes of the datasets

# %%
# Display the filtered train feature count
print("Filtered train features: {}".format(pca_train_features_label3.shape))

# Display the filtered valid feature count
print("Filtered valid features: {}".format(pca_valid_features_label3.shape))

# Display the filtered test feature count
print("Filtered test features: {}".format(pca_test_features_label3.shape))

# %%
pca_train_features_label3 = pca_train_features_label3.to_numpy()
train_label3 = train_label3.to_numpy()

pca_valid_features_label3 = pca_valid_features_label3.to_numpy()
valid_label3 = valid_label3.to_numpy()

pca_test_features_label3 = pca_test_features_label3.to_numpy()

# %% [markdown]
# ## Hyperparameter tuning

# %% [markdown]
# Define parameters for random search

# %%
knn_grid_params = {
    'n_neighbors' : [3, 5, 7, 9, 11, 13],
    'weights' : ['uniform', 'distance'],
    'metric' : ['minkowski', 'euclidean', 'manhattan', 'hamming']
}


rf_grid_params = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
}

svm_grid_params = {
    'C': [0.1, 1, 10, 30, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'auto', 'scale'],
    'kernel': ['rbf', 'linear']
}

# %% [markdown]
# Execute random search for KNN, Random forest and SVM

# %%
classification_models_params = [
    ('K Neighbors', KNeighborsClassifier(), knn_grid_params),
    ('Random Forest', RandomForestClassifier(), rf_grid_params),
    ('SVM', SVC(), svm_grid_params)
]

for model_name, model, grid_params in classification_models_params:
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rs = RandomizedSearchCV(
        estimator = model,
        param_distributions = grid_params,
        n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1
    )
    rs_result = rs.fit(pca_train_features_label3, train_label3)

    print(f"best score for {model_name} : {rs_result.best_score_}")
    print(f"best hyper parameters for {model_name} : {rs_result.best_params_}")

# %% [markdown]
# ## Model Selection

# %% [markdown]
# Select the model that best predicts the valid and test datasets based on accuracy, precision and recall

# %%
# Define a list of classification models
classification_models = [
    # ('K Neighbors', KNeighborsClassifier(n_neighbors=7, metric='minkowski', weights='distance')),
    # ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC(C=10, gamma='scale', kernel='rbf', class_weight='balanced')),
]

# The best model is SVM then KNN then Random Forest

# Number of features used in PCA
num_features = pca_train_features_label3.shape[1]
print(f"Number of features: {num_features}\n")

# Train and evaluate each classification model
for model_name, model in classification_models:
    # Train the model on the training data
    model.fit(pca_train_features_label3, train_label3)

    # Predict on the train data
    y_pred_train_label3 = model.predict(pca_train_features_label3)

    # Calculate metrics for classification evaluation
    accuracy = accuracy_score(train_label3, y_pred_train_label3)
    precision = precision_score(train_label3, y_pred_train_label3, average='weighted', zero_division=1)
    recall = recall_score(train_label3, y_pred_train_label3, average='weighted')

    print(f"Metrics for {model_name} on train data:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print("\n")

    # Predict on the validation data
    y_pred_valid_label3 = model.predict(pca_valid_features_label3)

    # Calculate metrics for classification evaluation on validation data
    accuracy = accuracy_score(valid_label3, y_pred_valid_label3)
    precision = precision_score(valid_label3, y_pred_valid_label3, average='weighted', zero_division=1)
    recall = recall_score(valid_label3, y_pred_valid_label3, average='weighted')

    print(f"Metrics for {model_name} on validation data:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print("\n")

    # Predict on the test data
    y_pred_test_label3 = model.predict(pca_test_features_label3)


# %% [markdown]
# # Label 4

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# Predict label 4 with feature engineering steps and techniques.  
# Use feature selection based on correlation matrix and feature extraction based on PCA

# %% [markdown]
# ### Data cleaning

# %% [markdown]
# Process the data to remove null values for labels and estimate missing values in features

# %% [markdown]
# Drop the columns where there are null values for the lables in the training dataset

# %%
# Check for null values in train dataset
print(f"train_data_label4 shape before : {train_data_label4.shape}")
train_null_counts = train_data_label4.isnull().sum()
print(f"train null counts before : \n{train_null_counts}")

# Drop rows with null values in the target labels for train dataset
train_data_label4 = train_data_label4.dropna(subset=train_data_label4.columns[-1:], how='any')
print(f"train_data_label4 shape after : {train_data_label4.shape}")

# %% [markdown]
# Fill the null values in the features with their means in the train, valid and test datasets.

# %%
# Fill null values with mean in train dataset
train_data_label4 = train_data_label4.fillna(train_data_label4.mean())

# Fill null values with mean in valid dataset
valid_data_label4 = valid_data_label4.fillna(valid_data_label4.mean())

# Fill null values with mean in test dataset
test_data = test_data.fillna(test_data.mean())

# %% [markdown]
# Extract the features and the label

# %%
train_features_label4 = train_data_label4.iloc[:, :-1]
train_label4 = train_data_label4.iloc[:, -1]

valid_features_label4 = valid_data_label4.iloc[:, :-1]
valid_label4 = valid_data_label4.iloc[:, -1]

test_features_label4 = test_data

# %% [markdown]
# Visualize the distribution of the training label 4

# %%
# Plotting the distribution of train_label4
labels, counts = np.unique(train_label4, return_counts=True)

plt.figure(figsize=(22, 6))
plt.xticks(labels)
plt.bar(labels, counts)
plt.xlabel('Target Label 4')
plt.ylabel('Frequency')
plt.title('Distribution of Target Label 4')
plt.show()

# %% [markdown]
# ### Feature Standardization

# %% [markdown]
# Standardize the features of all datasets using Robust scaler beacause outliers are present

# %%
# Standardize the features
scaler = RobustScaler()
standardized_train_features_label4 = scaler.fit_transform(train_features_label4)
standardized_valid_features_label4 = scaler.transform(valid_features_label4)
standardized_test_features_label4 = scaler.transform(test_features_label4)

# %% [markdown]
# ### Feature Extraction

# %% [markdown]
# Extract can combine the features that are highly significant in predicting the label using Principal Componenet Analysis(PCA)  
# Extract the features that can explain the variance of the label to 95%  
# Display the resulting explained variances of each principal component  

# %%
variance_threshold = 0.95

# Apply PCA with the determined number of components
pca = PCA(n_components=variance_threshold, svd_solver='full')

pca_train_features_label4 = pca.fit_transform(standardized_train_features_label4)
pca_valid_features_label4 = pca.transform(standardized_valid_features_label4)
pca_test_features_label4 = pca.transform(standardized_test_features_label4)

# Explained variance ratio after dimensionality reduction
explained_variance_ratio_reduced = pca.explained_variance_ratio_

# Plot explained variance ratio
plt.figure(figsize=(18, 10))
plt.bar(range(1, pca_train_features_label4.shape[1] + 1), explained_variance_ratio_reduced)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio per Principal Component (Reduced)')
plt.show()

# Display the reduced train feature matrix
print("Reduced Train feature matrix shape: {}".format(pca_train_features_label4.shape))
# Display the reduced valid feature matrix
print("Reduced valid feature matrix shape: {}".format(pca_valid_features_label4.shape))
# Display the reduced test feature matrix
print("Reduced test feature matrix shape: {}".format(pca_test_features_label4.shape))

# %% [markdown]
# ### Feature Selection

# %% [markdown]
# Calculate the correlation matrix of the training data features

# %%
#Calculate the correlation matrix
correlation_matrix = pd.DataFrame(pca_train_features_label4).corr()

mask = np.triu(np.ones_like(correlation_matrix))

# Create a heatmap of the correlation matrix using seaborn
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, mask=mask)
plt.title("Correlation Matrix")
plt.show()

# %% [markdown]
# Identify the features that are highly correlated with each other using the traning dataset

# %%
# Set the threshold for correlation
correlation_threshold = 0.9

highly_correlated = set()

# Find highly correlated features
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated.add(colname)

print(highly_correlated)

# %% [markdown]
# Remove the previously identified highly correlated features from all the datasets

# %%
# Remove highly correlated features
pca_train_features_label4 = pd.DataFrame(pca_train_features_label4).drop(columns=highly_correlated)
pca_valid_features_label4 = pd.DataFrame(pca_valid_features_label4).drop(columns=highly_correlated)
pca_test_features_label4 = pd.DataFrame(pca_test_features_label4).drop(columns=highly_correlated)

# %% [markdown]
# Display the resulting feature shapes of the datasets

# %%
# Display the filtered train feature count
print("Filtered train features: {}".format(pca_train_features_label4.shape))

# Display the filtered valid feature count
print("Filtered valid features: {}".format(pca_valid_features_label4.shape))

# Display the filtered test feature count
print("Filtered test features: {}".format(pca_test_features_label4.shape))

# %% [markdown]
# Convert dataframe to numpy array

# %%
pca_train_features_label4 = pca_train_features_label4.to_numpy()
train_label4 = train_label4.to_numpy()

pca_valid_features_label4 = pca_valid_features_label4.to_numpy()
valid_label4 = valid_label4.to_numpy()

pca_test_features_label4 = pca_test_features_label4.to_numpy()

# %% [markdown]
# ## Hyperparameter tuning

# %% [markdown]
# Define parameters for random search

# %%
knn_grid_params = {
    'n_neighbors' : [3, 5, 7, 9, 11, 13],
    'weights' : ['uniform', 'distance'],
    'metric' : ['minkowski', 'euclidean', 'manhattan', 'hamming']
}

rf_grid_params = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
}

svm_grid_params = {
    'C': [0.1, 1, 10, 30, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'auto', 'scale'],
    'kernel': ['rbf', 'linear']
}

# %% [markdown]
# Execute random search for KNN, Random forest and SVM

# %%
classification_models_params = [
    ('K Neighbors', KNeighborsClassifier(), knn_grid_params),
    ('Random Forest', RandomForestClassifier(), rf_grid_params),
    ('SVM', SVC(), svm_grid_params)
]

for model_name, model, grid_params in classification_models_params:
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rs = RandomizedSearchCV(
        estimator = model,
        param_distributions = grid_params,
        n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1
    )
    rs_result = rs.fit(pca_train_features_label4, train_label4)

    print(f"best score for {model_name} : {rs_result.best_score_}")
    print(f"best hyper parameters for {model_name} : {rs_result.best_params_}")

# %% [markdown]
# ## Model Selection

# %% [markdown]
# Select the model that best predicts the valid and test datasets based on accuracy, precision and recall

# %%
# Define a list of classification models
classification_models = [
    # ('K Neighbors', KNeighborsClassifier()),
    # ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC(C=10, gamma='scale', kernel='rbf')),
]

# The best model is SVM then KNN then Random Forest

# Number of features used in PCA
num_features = pca_train_features_label4.shape[1]
print(f"Number of features: {num_features}\n")

# Train and evaluate each classification model
for model_name, model in classification_models:
    # Train the model on the training data
    model.fit(pca_train_features_label4, train_label4)

    # Predict on the train data
    y_pred_train_label4 = model.predict(pca_train_features_label4)

    # Calculate metrics for classification evaluation
    accuracy = accuracy_score(train_label4, y_pred_train_label4)
    precision = precision_score(train_label4, y_pred_train_label4, average='weighted', zero_division=1)
    recall = recall_score(train_label4, y_pred_train_label4, average='weighted')

    print(f"Metrics for {model_name} on train data:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print("\n")

    # Predict on the validation data
    y_pred_valid_label4 = model.predict(pca_valid_features_label4)

    # Calculate metrics for classification evaluation on validation data
    accuracy = accuracy_score(valid_label4, y_pred_valid_label4)
    precision = precision_score(valid_label4, y_pred_valid_label4, average='weighted', zero_division=1)
    recall = recall_score(valid_label4, y_pred_valid_label4, average='weighted')

    print(f"Metrics for {model_name} on validation data:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print("\n")

    # Predict on the test data
    y_pred_test_label4 = model.predict(pca_test_features_label4)


# %% [markdown]
# # Generate Output CSV

# %% [markdown]
# Define method to create the csv file

# %%
# define method to create the dataframe and save it as a csv file
def create_csv(ID, pred_label1, pred_label2, pred_label3, pred_label4, destination):
  df = pd.DataFrame()
  
  df.insert(loc=0, column='ID', value=ID)
  df.insert(loc=1, column='label_1', value=pred_label1)
  df.insert(loc=2, column='label_2', value=pred_label2)
  df.insert(loc=3, column='label_3', value=pred_label3)
  df.insert(loc=4, column='label_4', value=pred_label4)

  df.to_csv(destination, index=False)

# %% [markdown]
# Create CSV file

# %% [markdown]
# Convert dataframe to numpy array

# %%
destination = './../results/layer 8/L8_190429G.csv'

# create the csv output file
create_csv(ID, y_pred_test_label1, y_pred_test_label2, y_pred_test_label3, y_pred_test_label4, destination)


