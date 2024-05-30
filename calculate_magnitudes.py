#Extracts light curves from a set of Blender output images

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import math
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Process satellite parameters for TLE propagation.')
    parser.add_argument('--ref_file', required=True, help='File path to reference object image')
    parser.add_argument('--topo_data_file', required=True, help='Path to topocentric data')
    parser.add_argument('--lc_plot_file', required=True, help='Path to save light curve plot')
    parser.add_argument('--lc_track_file', required=True, help='Path to save light curve track')
    parser.add_argument('--frames_dir', required=True, help='Directory to save frame images')
    parser.add_argument('--meta_file', required=True, help='Path to save metadata')
    parser.add_argument('--cr', required=True, type=float, help='Coefficient of reflectivity')
    parser.add_argument('--scale', required=True, type=float, help='Scale factor for Blender simulation')
    
    return parser.parse_args()

def get_norm(meta_file):
    with open(meta_file, 'r') as f:
        lines = f.readlines()

    norm = float(lines[14].split(': ')[1].strip())

    return abs(norm)



if __name__ == "__main__":
    args = parse_args()
    
    ref_file = args.ref_file
    topo_data_file = args.topo_data_file
    lc_plot_file = args.lc_plot_file
    lc_track_file = args.lc_track_file
    frames_dir = args.frames_dir
    meta_file = args.meta_file
    cr = args.cr
    scale = args.scale

    # Equation parameters --------------------------------------------
    rho_tot = 0.2 #Polished metallic surfaces have a total diffuse reflectance < total specular reflectance
    C = 1062 #W/m^2
    t_ext = 1/24
    A = 1.0 #m^2
    dot_cam_norm = 1.0
    d_ref = 10000 #km
    dot_sun_norm = get_norm(meta_file)

    # ----------------------------------------------------------------


    # Set the output directory for the images
    with open(topo_data_file, 'r') as f:
        lines = f.readlines()

    mags = []

    # Get reference image pixel count
    #open reference image
    ref_img = Image.open(ref_file)
    pixels = list(ref_img.getdata())
    B_ref = sum(sum(pixel) for pixel in pixels)
    print('B_ref = ', B_ref)

    I_mb = -2.5*np.log10(B_ref/t_ext)
    F_sun = C*dot_sun_norm
    F = (F_sun*rho_tot*A*dot_cam_norm)/d_ref**2
    mff = -26.7-2.5*np.log10(F/C)
    print('dot_sun_norm = ', dot_sun_norm)
    print('F = ', F)
    print('mff = ', mff)
    print('I_mb = ', I_mb)

    files = sorted(os.listdir(frames_dir))
    topo_file_line = 1
    # Go through files and range values
    for file in sorted(files):
        img = Image.open(frames_dir + file)
        img = img.convert('RGB')

        #Get the pixel values as a list of tuples
        pixels = list(img.getdata())

        #Calculate the sum of pixel values
        B = sum(sum(pixel) for pixel in pixels)
        print('B = ', B)

        #Get range in km
        d_target = float(lines[topo_file_line].split(' ')[3]) #[m]
        d_target_km = d_target/1000
        
        #Calculate apparent magnitude
        m = mff - I_mb + (-2.5*np.log10(B/(t_ext*(d_target_km/d_ref)**2)))
        print('m = ', m)

        mags.append(m)
        topo_file_line += 1

    #Plot the light curve
    # plt.plot(mags)
    # plt.xlabel('Frame')
    # plt.ylabel('Apparent Magnitude')
    # plt.title('Light curve of cylinder rotating about body x-axis')
    # plt.savefig('cyl-x.png')