#Extracts light curves from a set of Blender output images

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import math
import os
import sys
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Process satellite parameters for TLE propagation.')
    parser.add_argument('--track_num', required=True, help='Track number')
    parser.add_argument('--regime', required=True, help='Attitude regime of the object')
    parser.add_argument('--ref_file', required=True, help='File path to reference object image')
    parser.add_argument('--topo_data_file', required=True, help='Path to topocentric data')
    parser.add_argument('--lc_plot_file', required=True, help='Path to save light curve plot')
    parser.add_argument('--lc_track_file', required=True, help='Path to save light curve track')
    parser.add_argument('--frames_dir', required=True, help='Directory to save frame images')
    parser.add_argument('--meta_file', required=True, help='Path to save metadata')
    parser.add_argument('--scale', required=True, type=float, help='Scale factor for orekit simulation')
    parser.add_argument('--obj', required=True, help='Name of object being simulated')
    parser.add_argument('--rho_tot', required=True, type=float, help='Total diffuse reflectance of object')
    parser.add_argument('--fps', required=True, type=int, help='Frames per second of the simulation')
    return parser.parse_args()

def get_norm(meta_file):
    with open(meta_file, 'r') as f:
        lines = f.readlines()

    norm = float(lines[14].split(': ')[1].strip())

    return abs(norm)

def get_m(img_file, m_ref, I_ref, t_exp, d_target_km, d_ref, scale, mags):
    try:
        img = Image.open(img_file)
        img = img.convert('RGB')

        #Get the pixel values as a list of tuples
        pixels = list(img.getdata())
        # print(len(pixels))
        # print(type(pixels[0]))
        # print(pixels[0])

        #Calculate the sum of pixel values
        B = sum(sum(pixel) for pixel in pixels)
        # B = sum(0.299 * r + 0.587 * g + 0.114 * b for r, g, b in pixels)
        # print('B = ', B)

        m = m_ref - I_ref + (-2.5*np.log10(B/(t_exp*((d_target_km/d_ref)/scale)**2)))

    #If error reading image - Set m to previous value
    except Image.UnidentifiedImageError:
        print(f"Error: Cannot identify image file {frames_dir + file}.")
        m = mags[-1]
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        m = mags[-1]

    return m, B

def add_noise(data, noise_level=0.01):
    noisy_data = [x + random.uniform(-noise_level, noise_level) for x in data]
    
    return noisy_data

if __name__ == "__main__":
    args = parse_args()
    
    track_num = args.track_num
    regime = args.regime
    ref_file = args.ref_file
    topo_data_file = args.topo_data_file
    lc_plot_file = args.lc_plot_file
    lc_track_file = args.lc_track_file
    frames_dir = args.frames_dir
    meta_file = args.meta_file
    scale = args.scale
    obj = args.obj
    rho_tot = args.rho_tot
    fps = args.fps

    # Equation parameters --------------------------------------------
    C = 1062 #W/m^2
    t_exp = 1/fps #s - maximum allowable exposure time
    A = 1.0 #blender units^2
    dot_cam_norm = 1.0
    d_ref = 100 #blender units
    dot_sun_norm = get_norm(meta_file)
    # ----------------------------------------------------------------

    # Read file containing range information
    with open(topo_data_file, 'r') as f:
        range_data = f.readlines()

    range_data = range_data[1:]
    mags = []
    pixel_intensitites = []
    

    # Get reference image pixel count
    #open reference image
    ref_img = Image.open(ref_file)
    pixels = list(ref_img.getdata())

    B_ref = sum(sum(pixel) for pixel in pixels)
    # B_ref = sum(0.299 * r + 0.587 * g + 0.114 * b for r, g, b, _ in pixels)

    I_ref = -2.5*np.log10(B_ref/t_exp)
    F_sun = C*dot_sun_norm
    F = (F_sun*rho_tot*A*dot_cam_norm)/d_ref**2
    m_ref = -26.7-2.5*np.log10(F/C)

    files = sorted(os.listdir(frames_dir))
    
    # Get start index of topo data
    with open(meta_file, 'r') as f:
        metadata = {}
        for line in f:
            if ':' in line:
                key, value = line.split(': ', 1)
                metadata[key] = value.strip('\n')
    #get start index
    topo_file_line = int(metadata['Position Start Index'])
    
    # Go through files and range values
    for file in files:  
        
        #Get range in km
        d_target = float(range_data[topo_file_line].split(' ')[3]) #blender units
        d_target_km = d_target*scale
        
        full_path = frames_dir + file
        m, B = get_m(full_path, m_ref, I_ref, t_exp, d_target_km, d_ref, scale, mags)

        mags.append(m)
        pixel_intensitites.append(B)
        topo_file_line += 1


    # Add noise to the magnitudes
    noisy_mag = add_noise(mags)

    # Create a figure with two subplots
    frames = list(range(1, len(mags) + 1))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the light curve - no noise
    axs[0].plot(frames, mags)
    axs[0].set_xlabel('Frame')
    axs[0].set_ylabel('Apparent Magnitude')
    axs[0].set_title(f'Track {track_num} - {obj} - {regime}')
    axs[0].grid(True)

    # Plot the light curve - with noise
    axs[1].plot(frames, noisy_mag)
    axs[1].set_xlabel('Frame')
    axs[1].set_ylabel('Apparent Magnitude')
    axs[1].set_title(f'Track {track_num} - {obj} - {regime} - noise added')
    axs[1].grid(True)

    # Adjust layout for better spacing
    plt.tight_layout(pad=2.0)

    # Save the individual plots if needed
    fig.savefig(lc_plot_file)   

    # plt.show()

    with open(meta_file, 'a') as f:
        f.write(f'Reference Pixel Intenity: {B_ref}\n')
        f.write(f'Total Diffusivity: {rho_tot}\n')
        f.write(f'Frames: {len(files)}\n')
        f.write(f'Frames per Second: {fps}\n')
        f.write(f'Light Curve Duration [s]: {len(files)/fps}\n')
        

    with open(lc_track_file, 'w') as f:
        f.write('Apparent_Magnitudes Noise_Added Pixel_Intensities\n')
        for index, mag in enumerate(mags):
            f.write(str(mag) + ' ' + str(noisy_mag[index]) + ' ' + str(pixel_intensitites[index]) + '\n')