import pandas as pd
import os

location = f'/home/anne/Desktop/new/track_files/'
antennas = ['antenna_1.stl', 'antenna_2.stl', 'antenna_3.stl']
busses = ['bus_1.stl', 'bus_2.stl', 'bus_3.stl']
cones = ['cone_1.stl', 'cone_2.stl', 'cone_3.stl']
panels = ['panel_1.stl', 'panel_2.stl', 'panel_3.stl']
rods = ['rod_1.stl', 'rod_2.stl', 'rod_3.stl'] 
track_dirs = os.listdir(location)
dataset_len = len(track_dirs)
freq = 5    #5 Hz sampling rate (1 measurement every 0.2 seconds)

count_diffuse = 0
count_specular = 0
count_solar = 0
count_stable = 0
count_tumbling = 0
count_antennas = 0
count_busses = 0
count_cones = 0
count_panels = 0
count_rods = 0
count_full = 0
count_1min = 0
count_30s = 0
skipped = []

for dir in sorted(track_dirs):
    '''
    Open the metadata file in each directory and extract
        -number frames rendered
        -attitude regime
        -material
        -object shape
    '''

    track_num = int(dir.split('_')[-1])
    meta_file = os.path.join(location,dir,'metadata.txt')
    pos_file = os.path.join(location,dir,f'lc_track_{track_num}.txt')
    try:
        lc_info = pd.read_csv(pos_file, sep=' ', header=0)
        mags = lc_info['Apparent_Magnitudes']
        lc_len = len(mags)
        if lc_len == 400:
            count_full += 1
        elif lc_len >= 300:
            count_1min += 1
        elif lc_len >= 150:
            count_30s += 1

        with open(meta_file, 'r') as f:
            metadata = {}
            for line in f:
                if ':' in line:
                    key, value = line.split(': ', 1)
                    metadata[key] = value.strip('\n')

        regime = metadata['Attitude Regime']
        material = metadata['Material']
        name = metadata['Object Name']

        if material == 'DIFFUSE':
            count_diffuse += 1
        elif material == 'SPECULAR':
            count_specular += 1
        else:
            count_solar += 1

        if regime == 'STABLE':
            count_stable += 1
        else:
            count_tumbling += 1

        if name in antennas:
            count_antennas += 1
        elif name in busses:
            count_busses += 1
        elif name in cones:
            count_cones += 1
        elif name in panels:
            count_panels += 1
        else:
            count_rods += 1
  
    except:
        print(f'Issue reading track_{track_num} files') 
        skipped.append(track_num)
        
print('--------------------------------------')
print(f'Full (1min20s): {count_full}')
print(f'Partial (>1min): {count_1min}')
print(f'Short (>30s): {count_30s}')
print(f'Incomplete (<30s): {dataset_len - count_full - count_1min - count_30s}')
print(f'Total LCs: {dataset_len}')
print(f'Skipped Tracks: {skipped}')
print('--------------------------------------')
print(f'Diffuse: {count_diffuse}')
print(f'Specular: {count_specular}')
print(f'Solar: {count_solar}')
print('--------------------------------------')
print(f'Stable: {count_stable}')
print(f'Tumbling: {count_tumbling}')
print('--------------------------------------')
print(f'Antennas: {count_antennas}')
print(f'Busses: {count_busses}')
print(f'Cones: {count_cones}')
print(f'Panels: {count_panels}')
print(f'Rods: {count_rods}')