'''
Script for gathering all light curves into a numpy array and saving it to a file
the array stacks the light curves such that each row is one light curve and each column is a timestep
'''

import numpy as np
import os
import pandas as pd
import sys

def normalize(data):
    # Take mean and std across all data points
    mean = np.mean(data)
    std = np.std(data)

    return (data - mean) / std

length = 300
att_type = 'tumbling'
buffer = 50 # How much shorter the light curve can be than the desired length
save_location = '/home/anne/scripts/obj_sim/stacked_data/'
data_folder = f'/home/anne/Desktop/new/track_files/'
track_dirs = os.listdir(data_folder)
all_data = []

count = 0 #<---- remove
# Initialize an empty DataFrame with specified columns
catalogue = pd.DataFrame(columns=['Track', 'Object', 'Material', 'Regime', 'X_Rate', 'Y_Rate', 'Z_Rate' ,'Original_Length','Pad'])
for dir in sorted(track_dirs):
    track_num = int(dir.split('_')[1])
    if track_num != 8452 and track_num != 8549:
        location = f'/home/anne/Desktop/new/track_files/track_{track_num}/'
        lc_file = location + f'lc_track_{track_num}.txt'
        data= np.loadtxt(lc_file, delimiter=' ', skiprows=1)
        lc = data[:,0] # first column - no noise added
        og_len = len(lc)

        # Check rotation rates
        meta_file = location + 'metadata.txt'
        with open(meta_file, 'r') as f:
            metadata = {}
            for line in f:
                if ':' in line:
                    key, value = line.split(': ', 1)
                    metadata[key] = value.strip('\n')
        spin = []
        spin.append(float(metadata['Spin Rates [rad/s]'].split(', ')[0].split(' ')[1]))
        spin.append(float(metadata['Spin Rates [rad/s]'].split(', ')[1].split(' ')[1]))
        spin.append(float(metadata['Spin Rates [rad/s]'].split(', ')[2].split(' ')[1]))
        count_axes = 0
        for i in range(3):
            if spin[i] > 0.0:
                count_axes += 1

        # Includes the light curve based on length and number of spin axes
        if og_len >= length-buffer and count_axes >= 2:
            count+=1
            lc_array = np.array(lc)

            # Truncate longer light curves
            if len(lc_array) > length:
                lc_array = lc_array[:length]
                
            # Pad shorter light curves
            elif len(lc_array) < length:
                lc_array = np.pad(lc_array, pad_width=(0, length-len(lc_array)), mode='constant', constant_values=np.mean(lc_array))

            all_data.append(lc_array)
            
            obj_name = metadata['Object Name']
            regime = metadata['Attitude Regime']
            material = metadata['Material']
            n_frames = int(metadata['Frames'])
                
            data = {'Track': track_num,
                    'Object': obj_name,
                    'Material': material,
                    'Regime': regime,
                    'X_Rate': round(spin[0],3),
                    'Y_Rate': round(spin[1],3),
                    'Z_Rate': round(spin[2],3),
                    'Original_Length': og_len,
                    'Pad': round((length-og_len)/length,3)
                   }
            
            catalogue.loc[len(catalogue)] = data
            
        # if count == 2:
        #     break

data_stack = np.array(all_data)
data_stack = normalize(data_stack)
print(data_stack.shape)
print(catalogue)


catalogue.to_csv(save_location + f'catalogue_{length}_{att_type}.txt', sep=' ', index=False)
np.save(save_location + f'lc_stack_{length}_{att_type}.npy', data_stack)


