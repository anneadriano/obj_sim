'''
Performs tsne on tumbling class
shapes labelled on plot
        'antenna': 0,
        'bus': 1,
        'cone': 2,
        'panel': 3,
        'rod': 4
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import random
import sys

random.seed(42)

def build_dataset(catalogue, all_data):
    '''
    Takes the data in directly from the files of stacked data and returns
    the data and labels in separate X and y arrays
    '''

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
    
    # 4 classes only
    # class_data = {
    #     'antenna': [],
    #     'rect': [],
    #     'cone': [],
    #     'rod': [],
    # }

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
        'antenna': 0,
        'bus': 1,
        'cone': 2,
        'panel': 3,
        'rod': 4
    }

    # labels = {
    #     'antenna': 0,
    #     'rect': 1,
    #     'cone': 2,
    #     'rod': 3,
    # }


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

    # Define the colors for each category
    colors = ['blue', 'red', 'cyan', 'magenta', 'orange']
    cmap = ListedColormap(colors[:len(np.unique(y))])

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(data_tsne[:,0], data_tsne[:,1], data_tsne[:,2], c=y, cmap=cmap, edgecolor='k', alpha=0.7)

    # Create custom legend handles
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=5, label=label)
        for color, label in zip(colors, ['dish', 'cuboid', 'cone', 'panel', 'rod'])
    ]

    ax.legend(handles=handles, loc='best')
    ax.set_title('3D t-SNE Embedding')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    
    return data_tsne.reshape(data_tsne.shape[0], data_tsne.shape[1], 1)
    


# Input Parameters -----------------------------------------------------------
track_length = 300
att_type = 'tumbling'
test_size = 0.2

lc_stack_folder = '/home/anne/scripts/obj_sim/stacked_data/'
coeff_folder = '/home/anne/scripts/obj_sim/coefficients/J4Q16/'
catalogue_path = lc_stack_folder + f'catalogue_{track_length}_{att_type}.txt'

# data_path = lc_stack_folder + f'lc_stack_{track_length}_{att_type}.npy' #<--- For raw light curves
data_path = coeff_folder + f'Sx_{track_length}_{att_type}_allCoeffs.npy' #<--- For all scattering coefficients
# data_path = coeff_folder + f'Sx_{track_length}_{att_type}_order1Coeffs.npy' #<--- For 1st order scattering coefficients

catalogue = pd.read_csv(catalogue_path, sep=' ', header=0)
all_data = np.load(data_path)
print(all_data.shape)

X, y = build_dataset(catalogue, all_data)

print(X.shape)
print(y.shape)

# X = pca_reduction(X, 1)

X = tsne_reduction(X, y)
plt.show()