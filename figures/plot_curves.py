'''
Plots all trypes of curves for a single track

'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def plots(phases, ranges, mags, frames, track_num):
    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))

    # Plot the light curve - no noise
    axs[0,0].plot(frames, mags)
    # axs[0,0].scatter(frames, mags, c='black', s=2)
    axs[0,0].set_xlabel('Frame')
    axs[0,0].set_ylabel('Apparent Magnitude')
    axs[0,0].set_title(f'Track {track_num} - Mag vs. Frame')
    axs[0,0].invert_yaxis()
    axs[0,0].grid(True)

    # Plot mags against phase
    axs[0,1].plot(phases[:len(mags)], mags)
    axs[0,1].set_xlabel('Phase [deg]')
    axs[0,1].set_ylabel('Apparent Magnitude')
    axs[0,1].set_title(f'Track {track_num} - Mag vs. Phase')
    axs[0,1].invert_yaxis()
    axs[0,1].grid(True)

    # Plot phase vs. frame
    axs[1, 0].plot(frames, phases[:len(mags)])
    axs[1, 0].set_xlabel('Frame')
    axs[1, 0].set_ylabel('Phase [deg]')
    axs[1, 0].set_title(f'Track {track_num} - Phase vs. Frame')
    axs[1, 0].grid(True)

    # Placeholder plot (e.g., frame vs. frame)
    axs[1, 1].plot(frames, ranges[:len(mags)])
    axs[1, 1].set_xlabel('Frame')
    axs[1, 1].set_ylabel('Range [m]')
    axs[1, 1].set_title(f'Track {track_num} - Range vs. Frame')
    axs[1, 1].grid(True)

    # Show the plots
    plt.tight_layout()

    return

def draw_orbit(data, full_prop, start_index, n_frames, gs_coord):

    x = [float(line.split(', ')[0].strip('(')) for line in data]
    y = [float(line.split(', ')[1])  for line in data]
    z = [float(line.split(', ')[2].strip(')\n'))  for line in data]

    X = [float(line.split(', ')[0].strip('(')) for line in full_prop]
    Y = [float(line.split(', ')[1])  for line in full_prop]
    Z = [float(line.split(', ')[2].strip(')\n'))  for line in full_prop]

    gs_X = float(gs_coord.split(', ')[0].strip('('))
    gs_Y = float(gs_coord.split(', ')[1])
    gs_Z = float(gs_coord.split(', ')[2].strip(')\n'))

    # get range of visible positions
    end = start_index + n_frames
    vis_x = x[start_index:end]
    vis_y = y[start_index:end]
    vis_z = z[start_index:end]
    
    #Earth sphere
    # Convert spherical coordinates to Cartesian coordinates
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    radius = 63.78
    earth_x = radius * np.sin(phi) * np.cos(theta)
    earth_y = radius * np.sin(phi) * np.sin(theta)
    earth_z = radius * np.cos(phi)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    # Plot the data
    ax.plot_surface(earth_x, earth_y, earth_z, color='grey', alpha=0.2, rstride=10, cstride=10)
    ax.scatter(0, 0, 0, c='r', marker='+', label='Inertial Origin')
    ax.scatter(gs_X, gs_Y, gs_Z, c='b', marker='o', label='Ground Station')
    ax.plot(X, Y, Z, c='g', label='Full Propagation')
    ax.plot(x, y, z, c='pink', label='Visible Positions')
    ax.plot(vis_x, vis_y, vis_z, c='r', label='Light Curve Track Range')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of Object Positions (Units in $10^{-2}$ km)')
    ax.legend()

    return

def plot_pixels(pixels, frames):
    plt.figure()
    plt.plot(frames, pixels)
    plt.xlabel('Frame')
    plt.ylabel('Pixel Intensity')
    plt.title('Pixel Intensity vs. Frame')
    plt.tight_layout()

    return

def plot_mags(mags, frames, fps):
    t = [frame/fps for frame in frames]

    plt.figure()
    plt.plot(t, mags)
    plt.xlabel('Time [s]')
    plt.ylabel('Apparent Magnitude')
    plt.title('Apparent Magnitude vs.Time')
    plt.gca().invert_yaxis()  
    plt.tight_layout()

    return

# ------------------------------------------------------------------------------
track_num = 1952
freq = 5
# ------------------------------------------------------------------------------

location = f'/home/anne/Desktop/new/track_files/track_{track_num}/'
pos_file = location + 'obj_positions.txt'
full_pos_file = location + 'full_obj_positions.txt'
meta_file = location + 'metadata.txt'
lc_file = location + f'lc_track_{track_num}.txt'
phase_file =  location + 'topodata.txt'

#Bring position coordinates into lists
with open(pos_file, 'r') as f:
    vis_positions = f.readlines()

with open(full_pos_file, 'r') as f:
    all_positions = f.readlines()

#Create dictionary for metadata
with open(meta_file, 'r') as f:
    metadata = {}
    for line in f:
        if ':' in line:
            key, value = line.split(': ', 1)
            metadata[key] = value.strip('\n')

gs_coord = metadata['Ground Station Position']
start_index = int(metadata['Position Start Index'])
n_frames = int(metadata['Frames'])

# Bring the data into dataframes
df_lc = pd.read_csv(lc_file, sep=' ', header=0)
df_topo = pd.read_csv(phase_file, sep=' ', header=0)

phases = df_topo['Phase[deg]']
ranges = df_topo['Range[m]']
mags = df_lc['Apparent_Magnitudes']
pixels = df_lc['Pixel_Intensities']
frame_range = range(len(mags))

# plots(phases, ranges, mags, frame_range, track_num)
draw_orbit(vis_positions, all_positions, start_index, n_frames, gs_coord)
# plot_pixels(pixels, frame_range)
# plot_mags(mags, frame_range, freq)

plt.show()