'''
Analyzes only the usable data in the dataset
looks at the stack catalogue data for all regimes
'''

import numpy as np
import pandas as pd
import sys

track_length = 300
catalogue_path = f'/home/anne/scripts/obj_sim/stacked_data/catalogue_{track_length}_allRegimes.txt'
data_path = '/home/anne/Desktop/new/track_files/'
catalogue = pd.read_csv(catalogue_path, sep=' ', header=0)
print(catalogue.head())
num_lcs = len(catalogue)

stable_count = (catalogue['Regime'] == 'STABLE').sum()
tumbling_count = (catalogue['Regime'] == 'TUMBLING').sum()
cone_count = catalogue['Object'].str.contains('cone').sum()
dish_count = catalogue['Object'].str.contains('antenna').sum()
panel_count = catalogue['Object'].str.contains('panel').sum()
cuboid_count = catalogue['Object'].str.contains('bus').sum()
rod_count = catalogue['Object'].str.contains('rod').sum()

# Get avg comp time
comp_times = []

for track_num in catalogue['Track']:
    try:
        with open(data_path + f'track_{track_num}/metadata.txt', 'r') as f:
            for line in f:
                if 'Computation Time' in line:
                    comp_times.append(float(line.split(' ')[2]))
    except FileNotFoundError:
        print(f'File not found for track {track_num}')
        continue

avg_comp_time = np.mean(comp_times)

print(f'Average computation time: {avg_comp_time}')
print(f'Number of light curves: {num_lcs}')
print(f'Number of stable light curves: {stable_count}')
print(f'Number of tumbling light curves: {tumbling_count}')
print(f'Number of cone light curves: {cone_count}')
print(f'Number of dish light curves: {dish_count}')
print(f'Number of panel light curves: {panel_count}')
print(f'Number of cuboid light curves: {cuboid_count}')
print(f'Number of rod light curves: {rod_count}')


