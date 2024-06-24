'''
Requires track numbers and outputs the light curves (on the same plot) of the specified tracks
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
    lc = data[:,1]

    return lc

def plot_curves(array, freq, tracks):
    '''
    plots the light curves of the specified tracks
    '''

    num_tracks = array.shape[0]
    num_measurements = array.shape[1]
    num_plot = 400 # number of measurements to plot
    t = [frame/freq for frame in range(num_measurements)]

    plt.figure(figsize=(12, 5))
    for index, row in enumerate(array):
        plt.plot(t[:num_plot], row[:num_plot], label=f'track_{tracks[index]}')

    plt.xlabel('Time [s]')
    plt.ylabel('Apparent Magnitude')
    plt.title(f'Apparent Magnitude vs.Time')
    plt.gca().invert_yaxis()  
    plt.legend()
    plt.tight_layout()

#Inputs
#------------------------------------------------------------
tracks = [570, 593, 267]
num_plot = 400
freq = 5
#------------------------------------------------------------

all_data = []

for track in tracks:
    lc = get_curve(track)
    all_data.append(lc)

array_data = np.array(all_data)
mean = np.mean(array_data)
std = np.std(array_data)
norm_data = (array_data - mean) / std

plot_curves(norm_data, freq, tracks)
plt.show()