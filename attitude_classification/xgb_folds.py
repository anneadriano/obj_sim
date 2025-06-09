'''
Performs xgboost shape classification (no cross validation)
0 - STABLE
1 - TUMBLING
python attitude_classification/xgb_folds.py 2>&1 | tee attitude_classification/raw/classif_results.log
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import xgboost as xgb
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, SubsetRandomSampler
import random
import seaborn as sns
import time
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


def pca_reduction(X, components):
    samples = X.shape[0]
    x_dim = X.shape[2] #time resolution
    y_dim = X.shape[1] #frequency resolution

    #initialize array to hold pca results
    data_pca = np.zeros((samples, x_dim, 1))

    for i in range(x_dim):
        #extract ith column (time step) across all samples
        column_data = X[:, :, i]
        
        #apply pca to the extracted column data
        pca = PCA(n_components=components)
        principal_components = pca.fit_transform(column_data)

        #store the principal components in the data_pca array
        data_pca[:, i, :] = principal_components

    print(data_pca.shape)

    return data_pca

def tsne_reduction(X, y):
    #flatten the data
    samples = X.shape[0]
    X_flat = X.reshape(samples, -1)
    y=y.astype(int)

    # perform 3d tsne
    tsne = TSNE(n_components=3, random_state=42, metric='euclidean', perplexity=30)
    data_tsne = tsne.fit_transform(X_flat)

    print('Shape after tsne:', data_tsne.shape)

    #plot 3d tsne results
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=5, label='STABLE'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=5, label='TUMBLING')
    ]
    colors = ['b', 'r']
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(data_tsne[:,0], data_tsne[:,1], data_tsne[:,2], c=y, cmap='bwr', edgecolor='k', alpha=0.7)
    ax.legend(handles=handles, loc='best')
    ax.set_title('3D t-SNE Embedding')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    
    return data_tsne.reshape(data_tsne.shape[0], data_tsne.shape[1], 1)
    


# Input Parameters -----------------------------------------------------------
track_length = 300
test_size = 0.1
att_type = 'allRegimes'
lc_type = 'allCoeffs' #raw, coeffs, order1
tsne = True

params = {'objective':'binary:logistic',
        'eval_metric':'logloss',
        'booster':'gbtree',
        'learning_rate':0.001, #default 0.3
        'n_estimators':3000, #number of iterations of the algorithm
        'early_stopping_rounds': 200, #How many rounds of training occur without improvement on the validation metric
        'base_score':0.1, #initial prediction
        'subsample':0.8, #percentage of data randomly sampled to grow each tree
        'min_child_weight':3, #Higher values prevent a node from being split if the child nodes' weights are too small
        'max_depth':12,
        'colsample_bylevel':0.8, #percentage of features randomly chosen to grow each tree (default 1)
        'colsample_bynode':0.5, #percentage of features randomly chosen to split each node (default 1)
        'gamma':0.0, #higher means fewer splits in the tree and can prevent overfitting
        'reg_alpha': 1.0, #L1 regularization [0, 0.1, 1, 10], adds (increases/decreases) weight to certain features, higher alpha = more zero-weight features
        'reg_lambda':0.1,  #L2 regularization [0, 0.1, 1, 10], prevents from fitting to noise in data
        'random_state':42
}

lc_stack_folder = '/home/anne/scripts/obj_sim/stacked_data/'
coeff_folder = '/home/anne/scripts/obj_sim/coefficients/J4Q16/'
catalogue_path = lc_stack_folder + f'catalogue_{track_length}_{att_type}.txt'

if tsne:
    save_conf = f'/home/anne/scripts/obj_sim/attitude_classification/{lc_type}_tsne/confMatrices/'
    save_loss = f'/home/anne/scripts/obj_sim/attitude_classification/{lc_type}_tsne/lossCurves/'
else:
    save_conf = f'/home/anne/scripts/obj_sim/attitude_classification/{lc_type}/confMatrices/'
    save_loss = f'/home/anne/scripts/obj_sim/attitude_classification/{lc_type}/lossCurves/'


data_path = lc_stack_folder + f'lc_stack_{track_length}_{att_type}.npy' #<--- For raw light curves
# data_path = coeff_folder + f'Sx_{track_length}_{att_type}_allCoeffs.npy' #<--- For all scattering coefficients
# data_path = coeff_folder + f'Sx_{track_length}_{att_type}_order1Coeffs.npy' #<--- For 1st order scattering coefficients

catalogue = pd.read_csv(catalogue_path, sep=' ', header=0)
all_data = np.load(data_path)
print(all_data.shape)

X, y = build_dataset(catalogue, all_data)

if tsne:
    #bringin tsne np array
    # X = tsne_reduction(X, y)
    np.load(lc_stack_folder + f'tsne_{track_length}_{att_type}_{lc_type}.npy')
    # X = np.load(f'/home/anne/scripts/obj_sim/stacked_data/tsne_{track_length}_{att_type}_{lc_type}.npy')

print(X.shape)
print(y.shape)

# Split the combined data and labels into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

#Flatten the data
X_test_flat = np.reshape(X_test, (X_test.shape[0], -1))
y_test_flat = y_test.reshape(-1,)

print('Test size:', X_test_flat.shape)

# Set up folds and data
data_tensor = torch.from_numpy(X_train).float()
labels_tensor = torch.from_numpy(y_train).long()
dataset = TensorDataset(data_tensor, labels_tensor)

kf = KFold(n_splits=3)
test_accs = []
val_accs = []
train_accs = []

start = time.time()

for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
    print(f'---------------------------------- Fold {fold + 1} ----------------------------------')

    X_train = data_tensor[train_idx].numpy()
    y_train = labels_tensor[train_idx].numpy()
    X_val = data_tensor[val_idx].numpy()
    y_val = labels_tensor[val_idx].numpy()

    X_val_flat = np.reshape(X_val, (X_val.shape[0], -1))
    X_train_flat= np.reshape(X_train, (X_train.shape[0], -1))
    y_train_flat = y_train.reshape(-1,)
    y_val_flat = y_val.reshape(-1,)

    print('Train size: ', X_train_flat.shape)
    print('Validation size:', X_val_flat.shape)

    # Train the model
    watchlist = [(X_train_flat, y_train), (X_val_flat, y_val)]
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train_flat, y_train, eval_set=watchlist, verbose=False)

    # Evaluate the model's performance on Test Data
    y_pred = clf.predict(X_test_flat)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Get training accuracy
    y_train_pred = clf.predict(X_train_flat)
    y_val_pred = clf.predict(X_val_flat)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    test_accs.append(accuracy)
    val_accs.append(val_accuracy)
    train_accs.append(train_accuracy)

    print("Test Accuracy:", accuracy)
    print("Validation Accuracy:", val_accuracy)
    print("Train Accuracy:", train_accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", classification_rep)
    results = clf.evals_result()

    # Extract log loss values from evals_result
    train_loss = results['validation_0']['logloss']
    val_loss = results['validation_1']['logloss']

    # Plotting Results
    plt.figure(figsize=(8, 4))
    plt.plot(train_loss, label='Training')
    plt.plot(val_loss, label='Validation')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Log Loss')
    plt.title(f'XGBoost Loss Curves for Binary Classification (Fold {fold+1})')
    plt.legend()
    plt.savefig(f'{save_loss}xgb_loss_fold{fold + 1}.png', dpi=400)

    classes = ['Stable', 'Tumbling']
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'XGBoost Attitude Classification Results (fold {fold+1})')#  (with t-SNE Reduction)')
    plt.savefig(f'{save_conf}xgb_conf_fold{fold + 1}.png', dpi=400)

end = time.time()

print('---------------------------------- 5-Fold Cross Validation Complete ----------------------------------')
print('Train Accuracies:')
print(train_accs)
print('Validation Accuracies:')
print(val_accs)
print('Test Accuracies:')
print(test_accs)

avg_test = np.mean(test_accs)
avg_val = np.mean(val_accs)
print(f'Average test accuracy: {avg_test}')
print(f'Average validation accuracy: {avg_val}')

print('5 Fold Cross Validation takes {:.2f} seconds to complete'.format(end - start))



'''
# Plot feature importance ------------------------------------------------
top = 50
feature_importances = clf.feature_importances_

top_indices = feature_importances.argsort()[-top:][::-1]
print(top_indices)

# Get top 10 feature importances
top_feature_importances = feature_importances[top_indices]

# Plot
plt.figure(figsize=(10, 5))
plt.bar(range(top), top_feature_importances)
plt.xticks(range(top), top_indices, rotation=45)  # Use feature indices as x-axis label
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title(f'Top {str(top)} Feature Importances')
plt.show()

plt.show()
'''