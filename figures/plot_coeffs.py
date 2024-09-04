'''
Plots spectrograms for a certain sample
Refer to catalguecorresponding to Sx file for the sample index 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot(lc, Sx, order1_coeffs, track_num):
    t = np.arange(len(lc))*0.2

    plt.figure(figsize=(7, 7)) #width, height
    plt.subplot(3, 1, 1)
    # plt.plot(t,lc)
    plt.plot(t, lc, c='black', marker='.', markersize=2, linestyle=':', linewidth=0.9, label=f'Track {track_num}')
    plt.title('Original signal (track_' + str(track_num) + ')')
    plt.xlabel('Time [s]')
    plt.ylabel('Magnitude (Normalized)')
    plt.gca().invert_yaxis()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.imshow(Sx[:order1_coeffs, :], aspect='auto', cmap='viridis')#, origin='lower')
    plt.ylabel('log Frequency')
    plt.xlabel('Time Scale')
    plt.title('First-order scattering (track_' + str(track_num) + ')')
    # plt.colorbar(label='Power')

    plt.subplot(3, 1, 3)
    plt.imshow(Sx[order1_coeffs:, :], aspect='auto', cmap='viridis')#, origin='lower')
    plt.ylabel('log Frequency')
    plt.xlabel('Time Scale')
    plt.title('Second-order scattering (track_' + str(track_num) + ')')
    # plt.colorbar(label='Power')

    plt.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.08, hspace=0.4)
    # Parameters: left, right, top, bottom, wspace, hspace

    plt.show()
    return


def plot_lc(lc):
    t = np.arange(len(lc))*0.2

    # Plot the data against the time axis
    plt.figure(figsize=(8, 5))
    plt.plot(t, lc)
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Plot with Time Axis')
    plt.grid(True)

# Choose a track number
track_num = 5236
order1_coeffs = 31
order2_coeffs = 42

# Define file paths
cat_path = '/home/anne/scripts/obj_sim/stacked_data/catalogue_300_allRegimes.txt'
stack_path = '/home/anne/scripts/obj_sim/stacked_data/lc_stack_300_allRegimes.npy'
Sx_path = '/home/anne/scripts/obj_sim/coefficients/J4Q16/Sx_300_allRegimes_allCoeffs.npy'
catalogue = pd.read_csv(cat_path, sep=' ', header=0)
stack = np.load(stack_path)
Sx_full = np.load(Sx_path)

row_index = catalogue.loc[catalogue['Track'] == track_num].index[0]

print(Sx_full.shape)
print(stack.shape)

lc = stack[row_index]
Sx = Sx_full[row_index]
print(Sx.shape)

plot(lc, Sx, order1_coeffs, track_num)


# Show the plot
# plot_lc(lc)
# plt.show()



