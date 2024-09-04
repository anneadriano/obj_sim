'''
Generates plots fro tumbling  and stable light curves (side by side or separate)
doesnt not z-normalize the data
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def get_curve(track_num):
    '''
    Get the light curve for a specified track number
    returns the magnitudes as a np array
    '''
    location = f'/home/anne/Desktop/new/track_files/track_{track_num}/'
    lc_file = location + f'lc_track_{track_num}.txt'
    data = np.loadtxt(lc_file, delimiter=' ', skiprows=1)
    lc = data[:,0]

    return lc

def plot_curves_single(array, freq, tracks, measurements):
    '''
    plots the light curves of the specified tracks
    '''

    num_tracks = array.shape[0]
    t = [frame/freq for frame in range(measurements)]

    plt.figure(figsize=(8, 5))
    for index, row in enumerate(array):
        # plt.plot(t[:num_plot], row[:num_plot], label=f'Track {tracks[index]}')
        plt.plot(t[:measurements], row[:measurements], c='black', marker='.', markersize=4, linestyle=':', linewidth=0.8, label=f'Track {tracks[index]}')


    plt.xlabel('Time [s]')
    plt.ylabel('Apparent Magnitude')
    plt.title(f'Light Curve of Tumbling Rod')
    plt.gca().invert_yaxis()  
    plt.legend()
    plt.tight_layout()

def plot_separate(array_stable, array_tumbling, freq, measurements):
    t = [frame/freq for frame in range(measurements)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    shape_order = ['Dish', 'Dish','Dish', 'Dish', 'Cone', 'Cone', 'Cone', 'Cone', 'Cuboid', 'Cuboid', 'Cuboid', 'Cuboid', 'Panel', 'Panel', 'Panel', 'Panel', 'Rod', 'Rod', 'Rod', 'Rod']
    colours = ['green', 'green', 'green', 'green', 'red', 'red', 'red', 'red', 'purple', 'purple', 'purple', 'purple', 'orange', 'orange', 'orange', 'orange', 'blue', 'blue', 'blue', 'blue'] 

    # Plot the light curves for criteria 1
    for shape, colour, row in zip(shape_order, colours, array_tumbling):
        axs[0].plot(t, row[:measurements], label=shape, color=colour)

    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Apparent Magnitude')
    axs[0].set_title(f'Light Curves of Tumbling Objects')
    axs[0].grid(True)

    # #Handle Labels
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())

    # Plot the light curve for criteria 2
    for shape, colour, row in zip(shape_order, colours, array_stable):
        axs[1].plot(t, row[:measurements], label=shape, color=colour)
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Apparent Magnitude')
    axs[1].set_title('Light Curves of Stable Objects')
    axs[1].invert_yaxis()  
    axs[1].grid(True)

    #Handle Labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Adjust layout for better spacing
    plt.tight_layout(pad=2.0)

def normalize(array_data, data):
    '''
    z-normalization
    '''
    array_data = np.array(data)
    mean = np.mean(array_data)
    std = np.std(array_data)
    norm_data = (array_data - mean) / std

    return norm_data
shape_order = ['Dish', 'Dish','Dish', 'Dish', 'Cone', 'Cone', 'Cone', 'Cone', 'Cuboid', 'Cuboid', 'Cuboid', 'Cuboid', 'Panel', 'Panel', 'Panel', 'Panel', 'Rod', 'Rod', 'Rod', 'Rod']
colours = ['green', 'green', 'green', 'green', 'red', 'red', 'red', 'red', 'purple', 'purple', 'purple', 'purple', 'orange', 'orange', 'orange', 'orange', 'blue', 'blue', 'blue', 'blue'] 

#Inputs
#------------------------------------------------------------
tracks_tumbling = [4666, 3414, 2786, 5374, 2979, 5720, 3041, 7515, 6107, 6258, 6593, 2504, 1919, 495, 4298, 1441, 5771, 7570, 1952, 4657]
tracks_stable = [3238, 1750, 1284, 5789, 5236, 4062, 624, 7493, 1022, 125, 60, 1889, 5383, 3874, 3869, 8294, 860, 3568, 5541, 3589]
# tracks_tumbling = [1952]
# tracks_stable = []
measurements = 300
freq = 5
#------------------------------------------------------------

data_tumbling = []
data_stable = []

for track in tracks_tumbling:
    lc = get_curve(track)
    data_tumbling.append(lc)
for track in tracks_stable:
    lc = get_curve(track)
    data_stable.append(lc)

plot_separate(np.array(data_stable), np.array(data_tumbling), freq, measurements)
# plot_curves_single(np.array(data_tumbling), freq, tracks_tumbling, measurements)



plt.show()