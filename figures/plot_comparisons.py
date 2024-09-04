'''
Requires characteristics to look for and number of light curves to plot
Compares two sets of random light curves that correspond to the criteria
Uses z-normalization to compare the light curves
Ouputs the track numbers and light curves (on the same plot) that match the criteria
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
random.seed(42)

def plot_separate(array1, array2, obj_req1, obj_req2, regime_req1, regime_req2, freq, matches1, matches2, spins1, spins2):
    num_tracks = array1.shape[0]
    num_measurments = array1.shape[1]
    t = [frame/freq for frame in range(num_measurments)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # Plot the light curves for criteria 1
    for index, row in enumerate(array1):
        print(index)
        axs[0].plot(t, row, label=f'track_{matches1[index]}_{spins1[index][0]:.1f}_{spins1[index][1]:.1f}_{spins1[index][2]:.1f}')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Apparent Magnitude')
    axs[0].set_title(f'{obj_req1} Light Curves ({num_tracks}) with {regime_req1} Attitude Regime')
    axs[0].grid(True)
    axs[0].legend()

    # Plot the light curve for criteria 2
    for index, row in enumerate(array2):
        axs[1].plot(t, row, label=f'track_{matches2[index]}_{spins2[index][0]:.1f}_{spins2[index][1]:.1f}_{spins2[index][2]:.1f}')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Apparent Magnitude')
    axs[1].set_title(f'{obj_req2} Light Curves ({num_tracks}) in {regime_req2} Attitude Regime')
    axs[1].invert_yaxis()  
    axs[1].grid(True)
    axs[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout(pad=2.0)


def plot_same(array1, array2, obj_req1, obj_req2, regime_req1, regime_req2, freq, matches1, matches2):
    '''
    Plots the the light curves from the different criteria on the same plot
    '''
    
    num_tracks = array1.shape[0]
    num_measurements = array1.shape[1]
    num_plot = 400 # number of measurements to plot
    t = [frame/freq for frame in range(num_measurements)]

    plt.figure(figsize=(12, 5))
    for index, row in enumerate(array1):
        plt.plot(t[:num_plot], row[:num_plot], label=f'track_{matches1[index]}')
    for index, row in enumerate(array2):
        plt.plot(t[:num_plot], row[:num_plot], label=f'track_{matches2[index]}')

    plt.xlabel('Time [s]')
    plt.ylabel('Apparent Magnitude')
    plt.title(f'Apparent Magnitude vs.Time (Orange-{obj_req1}-{regime_req1}, Blue-{obj_req2}-{regime_req2})')
    plt.gca().invert_yaxis()  
    plt.legend()
    plt.tight_layout()


#Inputs
#------------------------------------------------------------
obj_req1 = 'cone'
regime_req1 = 'STABLE'
obj_req2 = 'panel'
regime_req2 = 'STABLE'
num_req = 10
frames_req = 400

folder = f'/home/anne/Desktop/new/track_files/'
track_dirs = os.listdir(folder)
fps = 5

matches1 = []
matches2 = []
spins1 = []
spins2 = []
random.shuffle(sorted(track_dirs))
random.shuffle(track_dirs)

for dir in track_dirs:
    track_num = int(dir.split('_')[1])

    if track_num != 8452 and track_num != 8549:

        # print(f'Checking track {track_num}')
        location = f'/home/anne/Desktop/new/track_files/track_{track_num}/'
        meta_file = location + 'metadata.txt'
        
        with open(meta_file, 'r') as f:
            metadata = {}
            for line in f:
                if ':' in line:
                    key, value = line.split(': ', 1)
                    metadata[key] = value.strip('\n')
        spin = []
        obj_name = metadata['Object Name']
        regime = metadata['Attitude Regime']
        n_frames = int(metadata['Frames'])
        spin.append(float(metadata['Spin Rates [rad/s]'].split(', ')[0].split(' ')[1]))
        spin.append(float(metadata['Spin Rates [rad/s]'].split(', ')[1].split(' ')[1]))
        spin.append(float(metadata['Spin Rates [rad/s]'].split(', ')[2].split(' ')[1]))

        
        if obj_req1.lower() in obj_name or obj_req2.lower() in obj_name:
            #Checks for object 1 requirements
            if obj_req1.lower() in obj_name and regime_req1 == regime and frames_req == n_frames and len(matches1) < num_req:
                # print(f'Track {track_num} matches requirements for criteria 1')
                matches1.append(track_num)
                spins1.append(spin)
            if obj_req2.lower() in obj_name and regime_req2 == regime and frames_req == n_frames and len(matches2) < num_req:
                # print(f'Track {track_num} matches requirements for criteria 2')
                matches2.append(track_num)
                spins2.append(spin)
        
        if len(matches1) == num_req and len(matches2) == num_req:
            break

print(f'Criteria 1: {matches1}')
print(f'Criteria 2: {matches2}')

all_data1 = []
all_data2 = []

for track_num1, track_num2 in zip(matches1, matches2):
    location1 = f'/home/anne/Desktop/new/track_files/track_{track_num1}/'
    lc_file1 = location1 + f'lc_track_{track_num1}.txt'
    data1= np.loadtxt(lc_file1, delimiter=' ', skiprows=1)
    lc1 = data1[:,1]
    all_data1.append(lc1)

    location2 = f'/home/anne/Desktop/new/track_files/track_{track_num2}/'
    lc_file2 = location2 + f'lc_track_{track_num2}.txt'
    data2= np.loadtxt(lc_file2, delimiter=' ', skiprows=1)
    lc2 = data2[:,1]
    all_data2.append(lc2)

# Convert the list of columns to a NumPy array
array1 = np.array(all_data1)
array2 = np.array(all_data2)
print(array1.shape)

# Combine arrays to get overall mean and standard deviation
combined = np.vstack((array1, array2))
mean = np.mean(combined) # Axis=0 performs operation along columns (returns 400 means)
std = np.std(combined)

# z-normalization
array1_norm = (array1 - mean) / std
array2_norm = (array2 - mean) / std

plot_separate(array1, array2, obj_req1, obj_req2, regime_req1, regime_req2, fps, matches1, matches2, spins1, spins2)   

# plot_same(array1_norm, array2_norm, obj_req1, obj_req2, regime_req1, regime_req2, fps, matches1, matches2)

plt.show()














