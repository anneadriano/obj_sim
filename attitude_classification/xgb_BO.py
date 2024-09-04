'''
Performs xgboost shape classification (no cross validation)
0 - STABLE
1 - TUMBLING
Bayesian Optimization
'''
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import random
import sys
random.seed(42)

def build_dataset(catalogue, all_data):
    # Reads the catalogue file as a df and creates a np array of labels
    
    # Generate a permutation of indices
    permutation = np.random.permutation(len(all_data))

    # Shuffle the NumPy array using the permutation
    all_data = all_data[permutation]

    # Shuffle the DataFrame using the permutation
    catalogue = catalogue.iloc[permutation].reset_index(drop=True)
    
   # Dictionary to hold class data
    class_data = {
        'TUMBLING': [],
        'STABLE': []
    }

    # Classify data based on catalogue labels
    for index, row in enumerate(all_data):
        att = catalogue.iloc[index]['Regime']
        if att == 'STABLE':
            class_data['STABLE'].append(row)
        else:
            class_data['TUMBLING'].append(row)


    # Determine the size of the smallest class
    smallest_class_size = min(len(class_data[key]) for key in class_data)

    # Print the smallest class size
    print(f'Smallest class size: {smallest_class_size}')

    # Trim all classes to the size of the smallest class
    for key in class_data:
      class_data[key] = class_data[key][:smallest_class_size]

    # Convert lists to numpy arrays
    for key in class_data:
      class_data[key] = np.array(class_data[key])

    # Create labels for each class
    labels = {
        'STABLE': 0,
        'TUMBLING': 1,
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
track_length = 300
att_type = 'allRegimes'
test_size = 0.2
num_folds = 5
highest_acc = 0.0

lc_stack_folder = '/home/anne/scripts/obj_sim/stacked_data/'
coeff_folder = '/home/anne/scripts/obj_sim/coefficients/J4Q16/'
catalogue_path = lc_stack_folder + f'catalogue_{track_length}_{att_type}.txt'

data_path = lc_stack_folder + f'lc_stack_{track_length}_{att_type}.npy' #<--- For raw light curves
# data_path = coeff_folder + f'Sx_{track_length}_{att_type}_allCoeffs.npy' #<--- For all scattering coefficients
# data_path = coeff_folder + f'Sx_{track_length}_{att_type}_order1Coeffs.npy' #<--- For 1st order scattering coefficients

catalogue = pd.read_csv(catalogue_path, sep=' ', header=0)
all_data = np.load(data_path)
print(all_data.shape)

X, y = build_dataset(catalogue, all_data)

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

print('Full train size: ', X_train_flat.shape)
print('Validation size:', X_val_flat.shape)
print('Test size:', X_test_flat.shape)

def objective(learning_rate, base_score, subsample, max_depth, reg_alpha, reg_lambda, X_combined=X_train_flat, y_combined=y_train_flat, num_folds=num_folds):
    '''
    Objective function for Bayesian Optimization
    '''
    params = {'objective':'binary:logistic',
        'eval_metric':'logloss',
        'booster':'gbtree',
        'n_estimators': 200, #number of iterations of the algorithm
        'early_stopping_rounds': 5, #How many rounds of training occur without improvement on the validation metric
        'random_state':42,
        'learning_rate': learning_rate, #0.1, #default 0.3
        'base_score': base_score, #0.5, #initial prediction
        'subsample':subsample,#0.8, #percentage of data randomly sampled to grow each tree
        # 'min_child_weight':3, #Higher values prevent a node from being split if the child nodes' weights are too small
        'max_depth':int(max_depth), #10,
        # 'colsample_bylevel':0.8, #percentage of features randomly chosen to grow each tree (default 1)
        # 'colsample_bynode':0.5, #percentage of features randomly chosen to split each node (default 1)
        # 'gamma':0, #higher means fewer splits in the tree and can prevent overfitting
        'reg_alpha':reg_alpha, #10 #L1 regularization [0, 0.1, 1, 10], adds (increases/decreases) weight to certain features, higher alpha = more zero-weight features
        'reg_lambda':reg_lambda #10,  #L2 regularization [0, 0.1, 1, 10], prevents from fitting to noise in data
    }

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # watchlist = [(X_train, y_train), (X_val, y_val)]
    # cv_results_loss = []
    cv_results_acc = [] 

    # Perform k-fold cross-validation
    for train_index, val_index in kf.split(X_combined):
        # Split combined data into training and validation sets
        X_train_cv, X_val_cv = X_combined[train_index], X_combined[val_index]
        y_train_cv, y_val_cv = y_combined[train_index], y_combined[val_index]

        # Set up watchlist for evaluation
        watchlist = [(X_train_cv, y_train_cv), (X_val_cv, y_val_cv)]

        # Initialize XGBoost classifier
        clf = xgb.XGBClassifier(**params)

        # Train the model
        clf.fit(X_train_cv, y_train_cv, eval_set=watchlist, verbose=False)

        # Evaluate the model on the validation set
        # eval_results = clf.evals_result()
        # min_loss = np.min(eval_results['validation_1']['logloss']) # logloss on validation set
        # cv_results_loss.append(min_loss)  

        # Get accuracy on test set
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_results_acc.append(accuracy)

        # if accuracy > highest_acc:
        #     highest_acc = accuracy
        #     confusion = confusion_matrix(y_test, y_pred)
        #     report = classification_report(y_test, y_pred)
        #     results = clf.evals_result()

    mean_acc = np.mean(cv_results_acc)

    return mean_acc

# Define bounds for parameters
pbounds = {
  'learning_rate': (0.0001, 0.1),
  'base_score': (0.2, 0.8),
  'subsample': (0.3, 0.8),
  'max_depth': (3, 15),
  'reg_alpha': (0.0, 10.1),
  'reg_lambda': (0.0, 10.1)
}

# Perform Bayesian optimization
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    verbose=2,
    random_state=42,
)


start = time.time()
optimizer.maximize(init_points=15, n_iter=15)
end = time.time()
print('Bayes optimization takes {:.2f} seconds to tune'.format(end - start))
print(optimizer.max)
