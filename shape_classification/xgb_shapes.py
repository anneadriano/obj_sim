'''
Performs xgboost shape classification (no cross validation)
0 - Dish
1 - Bus
2 - Cone
3 - Panel
4 - Rod
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import random
import sys

random.seed(42)

def build_dataset(catalogue, all_data, cls_size):
    # Reads the catalogue file as a df and creates a np array of labels
    
    # Generate a permutation of indices
    permutation = np.random.permutation(len(all_data))

    # Shuffle the NumPy array using the permutation
    all_data = all_data[permutation]

    # Shuffle the DataFrame using the permutation
    catalogue = catalogue.iloc[permutation].reset_index(drop=True)
    
   # Dictionary to hold class data
    class_data = {
        'antenna': [],
        'bus': [],
        'cone': [],
        'panel': [],
        'rod': []
    }

    # Classify data based on catalogue labels
    for index, row in enumerate(all_data):
        obj = catalogue.iloc[index]['Object']
        if 'antenna' in obj:
            class_data['antenna'].append(row)
        elif 'bus' in obj:
            class_data['bus'].append(row)
        elif 'cone' in obj:
            class_data['cone'].append(row)
        elif 'panel' in obj:
            class_data['panel'].append(row)
        else:
            class_data['rod'].append(row)


    # Ensure class sizes
    for key in class_data:
        # print(len(class_data[key]))
        class_data[key] = class_data[key][:cls_size]

    # Convert lists to numpy arrays
    for key in class_data:
        class_data[key] = np.array(class_data[key])

    # Create labels for each class
    labels = {
        'antenna': 0,
        'bus': 1,
        'cone': 2,
        'panel': 3,
        'rod': 4
    }
    class_labels = {key: np.full(len(class_data[key]), labels[key]) for key in class_data}

    # Combine the data and labels
    X_stack = np.concatenate(list(class_data.values()))
    y_stack = np.concatenate(list(class_labels.values()))

    # Reshuffle the data and labels
    permutation = np.random.permutation(len(X_stack))
    X = X_stack[permutation]
    y = y_stack[permutation]

    return X, y

# Input Parameters -----------------------------------------------------------
cls_size = 391
track_length = 300
test_size = 0.2

params = {'objective':'multi:softprob',
        'eval_metric':'mlogloss',
        'booster':'gbtree',        
        'num_class':5, #number of classes
        'learning_rate':0.1, #default 0.3
        'n_estimators':50, #number of iterations of the algorithm
        'early_stopping_rounds':5, #How many rounds of training occur without improvement on the validation metric
        'base_score':0.5, #initial prediction
        'subsample':0.3, #percentage of data randomly sampled to grow each tree
        'min_child_weight':3, #Higher values prevent a node from being split if the child nodes' weights are too small
        'max_depth':6,
        'colsample_bylevel':0.7, #percentage of features randomly chosen to grow each tree (default 1)
        'colsample_bynode':0.5, #percentage of features randomly chosen to split each node (default 1)
        'gamma':0, #higher means fewer splits in the tree and can prevent overfitting
        'reg_alpha':0.1, #L1 regularization [0, 0.1, 1, 10], adds (increases/decreases) weight to certain features, higher alpha = more zero-weight features
        'reg_lambda':0.1,  #L2 regularization [0, 0.1, 1, 10], prevents from fitting to noise in data
        'random_state':42
}

lc_stack_folder = '/home/anne/scripts/obj_sim/shape_classification/stacked_data/'
coeff_folder = '/home/anne/scripts/obj_sim/shape_classification/coefficients/J4Q16/'
catalogue_path = lc_stack_folder + f'catalogue_{track_length}_rotating.txt'

data_path = lc_stack_folder + f'lc_stack_{track_length}_rotating.npy' #<--- For raw light curves
# data_path = coeff_folder + f'Sx_{track_length}_rotating.npy' #<--- For scattering coefficients


catalogue = pd.read_csv(catalogue_path, sep=' ', header=0)
all_data = np.load(data_path)
print(all_data.shape)

X, y = build_dataset(catalogue, all_data, cls_size)

print(X.size)

# Split the combined data and labels into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42) 

#Flatten the data
X_train_flat= np.reshape(X_train, (X_train.shape[0], -1))
X_test_flat = np.reshape(X_test, (X_test.shape[0], -1))
X_val_flat = np.reshape(X_val, (X_val.shape[0], -1))
y_train_flat = y_train.reshape(-1,)
y_test_flat = y_test.reshape(-1,)
y_val_flat = y_val.reshape(-1,)

print('Train size: ', X_train_flat.shape)
print('Validation size:', X_val_flat.shape)
print('Test size:', X_test_flat.shape)

# Train the model
watchlist = [(X_train_flat, y_train), (X_val_flat, y_val)]
clf = xgb.XGBClassifier(**params)
clf.fit(X_train_flat, y_train, eval_set=watchlist, verbose=True)
y_pred = clf.predict(X_test_flat)

# Evaluate the model's performance on Validation Data
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_rep)
results = clf.evals_result()

# Extract log loss values from evals_result
train_loss = results['validation_0']['mlogloss']
test_loss = results['validation_1']['mlogloss']

#Plotting Results
plt.figure(figsize=(8, 4))
plt.plot(train_loss, label='Train')
plt.plot(test_loss, label='Test')
plt.xlabel('Boosting Rounds')
plt.ylabel('Log Loss')
plt.title('XGBoost Loss Curves for Binary Classification')
plt.legend()
plt.show()