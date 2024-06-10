import matplotlib.pyplot as plt
import pandas as pd


track_num = 24

lc_file = f'/home/anne/Desktop/track_files/track_{track_num}/lc_track_{track_num}.txt'
phase_file = f'/home/anne/Desktop/track_files/track_{track_num}/topodata.txt'

# Bring the data into dataframes
df_lc = pd.read_csv(lc_file, sep=' ', header=0)
df_topo = pd.read_csv(phase_file, sep=' ', header=0)

phases = df_topo['Phase[deg]']
ranges = df_topo['Range[m]']
mags = df_lc['Apparent_Magnitudes']
frames = range(len(mags))

print(mags.head())
print(phases.head())

# Create a figure with two subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot the light curve - no noise
axs[0,0].plot(frames, mags)
axs[0,0].set_xlabel('Frame')
axs[0,0].set_ylabel('Apparent Magnitude')
axs[0,0].set_title(f'Track {track_num} - Mag vs. Frame')
axs[0,0].grid(True)

# Plot mags against phase
axs[0,1].plot(phases[:len(mags)], mags)
axs[0,1].set_xlabel('Phase [deg]')
axs[0,1].set_ylabel('Apparent Magnitude')
axs[0,1].set_title(f'Track {track_num} - Mag vs. Phase')
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
plt.show()