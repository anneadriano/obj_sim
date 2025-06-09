'''
Performs t-SNE on the given data and saves the results to a file
python attitude_classification/tsne.py 2>&1 | tee attitude_classification/tsne_results/results_allRegimes_allCoeffs.log
'''

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time
import sys



def tsne_reduction(X, y, plot_save_loc):
    #flatten the data
    samples = X.shape[0]
    X_flat = X.reshape(samples, -1)
    y=y.astype(int)

    #--------------------------------------------
    # perform 3d tsne
    #--------------------------------------------
    tsne = TSNE(n_components=3, random_state=42, metric='euclidean', perplexity=30)
    data_tsne = tsne.fit_transform(X_flat)

    print('Shape after 3D tsne:', data_tsne.shape)

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

    plt.savefig(plot_save_loc + f'tsne3D_{att_type}_{coeffs}.png', dpi=400)
    
    return data_tsne.reshape(data_tsne.shape[0], data_tsne.shape[1], 1)

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

track_length = 300
att_type = 'allRegimes'
coeffs = 'allCoeffs' # 'order1Coeffs' or 'allCoeffs'

lc_stack_folder = '/home/anne/scripts/obj_sim/stacked_data/'
coeff_folder = '/home/anne/scripts/obj_sim/coefficients/J4Q16/'
catalogue_path = lc_stack_folder + f'catalogue_{track_length}_{att_type}.txt'
save_loc = '/home/anne/scripts/obj_sim/attitude_classification/tsne_results/'

if coeffs == 'raw':
    data_path = lc_stack_folder + f'lc_stack_{track_length}_{att_type}.npy' #<--- For raw light curves
elif coeffs == 'allCoeffs':
    data_path = coeff_folder + f'Sx_{track_length}_{att_type}_allCoeffs.npy' #<--- For all scattering coefficients
elif coeffs == 'order1Coeffs':  
    data_path = coeff_folder + f'Sx_{track_length}_{att_type}_order1Coeffs.npy' #<--- For 1st order scattering coefficients

catalogue = pd.read_csv(catalogue_path, sep=' ', header=0)
all_data = np.load(data_path)
print(all_data.shape)

X, y = build_dataset(catalogue, all_data)

print(X.shape)
print(y.shape)

start = time.time()

X = tsne_reduction(X, y, save_loc)

end = time.time()


print('Parameters:')
print(f'Track length: {track_length}, Attitude type: {att_type}, Coefficients: {coeffs}')
print(f'Time taken for t-SNE: {end - start} seconds')

# Save the t-SNE reduced data
np.save(lc_stack_folder + f'tsne_{track_length}_{att_type}_{coeffs}.npy', X)

