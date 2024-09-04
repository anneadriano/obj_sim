'''
Takes in a light curve track nuimber and plots the corresponding lomb-scargle periodogram
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_mags(mags, t):

    plt.figure()
    plt.plot(t, mags)
    plt.xlabel('Time [s]')
    plt.ylabel('Apparent Magnitude')
    plt.title('Apparent Magnitude vs.Time')
    plt.gca().invert_yaxis()  
    plt.tight_layout()

    return

def plot_ls(per, pow, track_num):
    plt.figure(figsize=(10, 6))
    plt.plot(per, pow)
    plt.xlabel('Period [s]')
    plt.ylabel('Power')
    plt.xscale('log')
    plt.title(f'Lomb-Scargle Periodogram of Track {track_num}')

def plot_inset(per, pow, track_num):

    fig, ax = plt.subplots(figsize=(10, 6))

    # Main plot with log-scaled x-axis
    ax.plot(per, pow)
    ax.set_xscale('log')
    ax.set_xlabel('Log-Scaled Period [s]')
    ax.set_ylabel('Power')
    ax.legend()
    ax.set_title(f'Lomb-Scargle Periodogram of Track {track_num}')

    # Create an inset plot
    inset_ax = inset_axes(ax, width="30%", height="30%", loc='upper right')

    # Define the region to zoom in
    x1, x2 = 1, 2  # X-axis range to zoom in on
    y1, y2 = -1, 1   # Y-axis range to zoom in on

    # Plot the zoomed-in section
    inset_ax.plot(per, pow)
    inset_ax.set_xlim(x1, x2)
    # inset_ax.set_ylim(y1, y2)
    inset_ax.set_title('Zoomed Inset')

    return

def plot_fft(freq, fft_result, track_num):
    plt.figure()
    plt.plot(freq[1:], fft_result[1:])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title(f'FFT of Track {track_num}')

    return


# track_num = 5221
track_num = 232
fps = 5
frames = 200

lc_file = f'/home/anne/Desktop/new/track_files/track_{track_num}/lc_track_{track_num}.txt'

lc_info = pd.read_csv(lc_file, sep=' ', header=0)
mags = lc_info['Apparent_Magnitudes']
mags_shortened = mags[:frames]
frame_range = range(len(mags_shortened))
t = [frame/fps for frame in frame_range]

frequency, power = LombScargle(t, mags_shortened).autopower(minimum_frequency=0.0, maximum_frequency=1.0)#, normalization='psd')
period = 1/frequency

# fft_result = np.fft.fft(mags_shortened)
# fft_freq = np.fft.fftfreq(len(t), 1/fps)
# positive_freqs = fft_freq[:len(fft_freq)//2]
# positive_fft_result = np.abs(fft_result[:len(fft_result)//2])

# plot_fft(positive_freqs, positive_fft_result, track_num)

# plot_ls(period, power, track_num)
# plot_inset(period, power, track_num)
# plot_mags(mags_shortened, np.arange(frames), fps)

plt.show()