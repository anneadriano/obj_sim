'''
Gets wavelet coefficients from light curve stack created by collect_lcs.py
Takes in the data in the form of.npy files and extracts wavelet scattering features
Saves Sx file with coefficients but does not save 0th order coefficients
'''
import numpy as np
from kymatio.numpy import Scattering1D

import matplotlib.pyplot as plt
import os

def feature_extract(data, J, Q): 
    '''
    data should be in the form (1,n_samples)
    returns scattering features
    '''
    
    T = data.shape[-1]
    scattering = Scattering1D(J=J, shape=T, Q=Q)#, output_type='list')
    Sx = scattering(data) #Output in the form (paths, coefficients, samples)
    meta = scattering.meta()

    num_coefficients = scattering.output_size(detail=True)
    num_order0_coeffs = num_coefficients[0]
    num_order1_coeffs = num_coefficients[1]
    num_order2_coeffs = num_coefficients[2]

    print('Number of coefficients:', num_coefficients)
    print('Number of zeroth-order coefficients:', num_order0_coeffs)
    print('Number of first-order coefficients:', num_order1_coeffs)
    print('Number of second-order coefficients:', num_order2_coeffs)

    return Sx, meta, num_order0_coeffs, num_order1_coeffs, num_order2_coeffs

def plot_row_of_stack(stack, row, Sx, n_coeffs_1):
    # idx = np.where(meta['order'] == row)
    
    plt.figure(figsize=(6, 7.5)) #width, height

    plt.subplot(4, 1, 1)
    plt.plot(np.arange(0, stack[row].shape[-1]*0.2, 0.2),stack[row].reshape(-1,1))
    plt.title('Original signal')
    plt.xlabel('Time [s]')
    plt.ylabel(' Apparent Magnitude')
    plt.grid(True)

    plt.subplot(4, 1, 2)
    # plt.plot(Sx[idx][0][0])
    plt.plot(Sx[row, 0, :])
    plt.title('Zeroth-order scattering')

    plt.subplot(4, 1, 3)
    # plt.imshow(Sx[idx][0][1:n_coeffs_1], aspect='auto', cmap='viridis')
    plt.imshow(Sx[row, 1:n_coeffs_1, :], aspect='auto', cmap='viridis')
    plt.title('First-order scattering')
    # plt.colorbar(label='Power')

    plt.subplot(4, 1, 4)
    # plt.imshow(Sx[idx][0][n_coeffs_1:], aspect='auto', cmap='viridis')
    plt.imshow(Sx[row, n_coeffs_1:, :], aspect='auto', cmap='viridis')    
    plt.title('Second-order scattering')
    # plt.colorbar(label='Power')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, hspace=0.7)
    # Parameters: left, right, top, bottom, wspace, hspace

    plt.show()

track_length = 300 #<-------CHANGE
att_type = 'tumbling' #<-------CHANGE
data_location = '/home/anne/scripts/obj_sim/stacked_data/'
save_location = '/home/anne/scripts/obj_sim/coefficients/J4Q16/' # <--------

file = f'lc_stack_{track_length}_{att_type}.npy'
J = 4
Q = 16

data = np.load(data_location + file)
print(data.shape)

Sx, meta, n_coeffs_0, n_coeffs_1, n_coeffs_2 = feature_extract(data, J, Q)
print(Sx.shape)
print(type(Sx))
print(Sx[:, 1:, :].shape)     

np.save(save_location + f'Sx_{track_length}_{att_type}_allCoeffs.npy', Sx[:, 1:, :]) #<-------CHANGE (does not save zeroth order coefficients)
np.save(save_location + f'Sx_{track_length}_{att_type}_order1Coeffs.npy', Sx[:, 1:n_coeffs_1, :]) #<-------CHANGE (does not save zeroth order coefficients)


#PLOTTING - OPTIONAL
# row_to_plot = 500 # Corresponds to row in catalogue file
# plot_row_of_stack(data, row_to_plot, Sx, n_coeffs_1) 

