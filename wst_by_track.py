''''
Creates the spectrograms of a light curve based on track number
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

def plot_row_of_stack(lc_array, row, Sx, n_coeffs_1):
    # idx = np.where(meta['order'] == row)
    
    plt.figure(figsize=(6, 7.5)) #width, height

    plt.subplot(4, 1, 1)
    plt.plot(np.arange(0, lc_array.shape[-1]*0.2, 0.2),lc_array.reshape(-1,1))
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

track = 1952
J = 4
Q = 16

data_location = f'/home/anne/Desktop/new/track_files/track_{track}/lc_track_{track}.txt'

# Bring the data into dataframe
data= np.loadtxt(data_location, delimiter=' ', skiprows=1)
lc = data[:,0]
lc_array = np.array(lc)
lc_array = lc_array.reshape(1, len(lc_array))
print(lc_array.shape)

Sx, meta, num_order0_coeffs, num_order1_coeffs, num_order2_coeffs = feature_extract(lc_array, J, Q)

print('Sx shape:', Sx.shape)
print('1st order:', num_order1_coeffs)
print('2nd order:', num_order2_coeffs)

plot_row_of_stack(lc_array, 0, Sx, num_order1_coeffs)
