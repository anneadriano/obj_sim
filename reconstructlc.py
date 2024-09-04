'''
Script to recreate a light curve produced by main.py
Takes in all metadata from specified track and regenerates light curve at different sampling frequency
'''

import subprocess 
import os
import sys
import random
from math import pi
import time

def run_command(cmd):
    try:
        # Start the subprocess
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Read the output line by line as it is produced
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())

        # Read the remaining stderr after the process ends
        stderr = process.stderr.read()
        if stderr:
            print(stderr.strip())

        # Check for any errors
        rc = process.poll()
        if rc != 0:
            raise subprocess.CalledProcessError(returncode=rc, cmd=cmd, output=stderr)
        
        return 0

    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e.output}")
        return 1
    except subprocess.TimeoutExpired:
        print("Command timed out")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    
def create_directory(dir_path):

    try:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory {dir_path} created successfully.")
    except Exception as e:
        print(f"Error creating directory {dir_path}: {e}")

    return

# INPUT PARAMETERS -----------------------------------------------------------
track_num = 1952
fps = 5 # Frames per second
timestep = 0.2 # Timestep in seconds for orbit propagation
n_frames = 400 # Number of frames to render
# ----------------------------------------------------------------------------

# Constants
scale_blender = 1/100
scale_orekit = 1/100000
so_bank = './objects/'
track_directory = f'/home/anne/Desktop/reconstruct/track_files/track_{track_num}_/'
frames_file_path = f'/home/anne/Desktop/reconstruct/frames/frames_{track_num}_/'
ref_file = track_directory + 'reference_image'
positions_file = track_directory + "obj_positions.txt"
full_positions_file = track_directory + "full_obj_positions.txt"
topo_data_file = track_directory + "topodata.txt"
lc_plot_file = track_directory + f"lc_plot_{track_num}.png"
lc_track_file = track_directory + f"lc_track_{track_num}.txt"
lc_plot_noise_file = track_directory + f"lc_plot_noise_{track_num}.png"
meta_new = track_directory + 'metadata.txt'
print(f"Reconstructing track {track_num}.")
create_directory(track_directory)
create_directory(frames_file_path)

# Read info from original light curve
og_directory = f'/home/anne/Desktop/new/track_files/track_{track_num}/'
meta_old = og_directory + "metadata.txt"
with open(meta_old, 'r') as f:
    metadata = {}
    for line in f:
        if ':' in line:
            key, value = line.split(': ', 1)
            metadata[key] = value.strip('\n')

spin = []
orientation = []
obj_name = metadata['Object Name']
regime = metadata['Attitude Regime']
n_frames = int(metadata['Frames'])
mass = float(metadata['Mass'])
cross_sect = float(metadata['Cross Section'])
cd = float(metadata['Coefficient of Drag'])
material = metadata['Material']
tle_0 = metadata['TLE Line 0']
spin.append(float(metadata['Spin Rates [rad/s]'].split(', ')[0].split(' ')[1]))
spin.append(float(metadata['Spin Rates [rad/s]'].split(', ')[1].split(' ')[1]))
spin.append(float(metadata['Spin Rates [rad/s]'].split(', ')[2].split(' ')[1]))

# Set up other parameters
obj_path = so_bank + obj_name
if material == 'DIFFUSE':
    colour = [0.6, 0.6, 0.6]
    roughness = 0.4
    metallic = 1.0
    rho_tot = 0.6
    ior = 1.5
    cr = 1.2
elif material == 'SPECULAR':
    colour = [0.6, 0.6, 0.6]
    roughness = 0.05
    metallic = 0.8
    rho_tot = 0.1
    ior = 2.0
    cr = 1.7
else: # SOLAR
    colour = [0.002, 0.001, 0.012]
    roughness = 0.1
    metallic = 0.8
    rho_tot = 0.6
    ior = 1.5
    cr = 0.5

#search tle files
tle_files = os.listdir('/home/anne/scripts/obj_sim/TLEs/')
for file in tle_files:
    if tle_0 in open(f'/home/anne/scripts/obj_sim/TLEs/{file}').read():
        tle_file = f'/home/anne/scripts/obj_sim/TLEs/{file}'
        break
code = 1
while code != 0:
    
    start_time = time.time()

    # Run orbit propagator ----------------------------------------------------
    cmd = [
            'python', 'tle_propagator.py', 
            '--track_num', str(track_num),
            '--track_dir', str(track_directory),
            '--obj_name', str(obj_name),
            '--obj_path', str(obj_path),
            '--tle_file', str(tle_file),
            '--positions_file', str(positions_file),
            '--full_positions_file', str(full_positions_file),
            '--meta_file', str(meta_new),
            '--topo_data_file', str(topo_data_file),
            '--scale', str(scale_orekit),
            '--regime', str(regime),
            '--spin_x', str(spin[0]),
            '--spin_y', str(spin[1]),
            '--spin_z', str(spin[2]),
            '--mass', str(mass), # Mass of satellite in kg
            '--cross_sect', str(cross_sect), # Cross section of satellite in m^2
            '--cd', str(cd), # Coefficient of drag
            '--cr', str(cr), # Coefficient of reflectivity
            '--timestep', str(timestep), # Timestep in seconds
            '--num_frames', str(n_frames)
        ]
    code = run_command(cmd)
    print(f'Code: {code}')


#Run blender simulation with object import --------------------------------
cmd = [
    'blender', '-P', 'blender_sim_noAO.py', '--',
    '--track_num', str(track_num),
    '--obj_path', str(obj_path),
    '--meta_file', str(meta_new),
    '--frames_dir', str(frames_file_path),
    '--positions_file', str(positions_file),
    '--scale', str(scale_blender),
    '--metallic', str(metallic),
    '--roughness', str(roughness),
    '--spin_x', str(spin[0]),
    '--spin_y', str(spin[1]),
    '--spin_z', str(spin[2]),
    '--n_frames', str(n_frames),
    '--fps', str(fps),
    '--material', str(material),
    '--r', str(colour[0]),
    '--g', str(colour[1]),
    '--b', str(colour[2]),
    '--ior', str(ior)
]
run_command(cmd)

# Run blender simulation for reference object -----------------------------
cmd = [
    'blender', '-P', 'blender_sim_ref.py', '--',
    '--track_num', str(track_num),
    '--ref_file', str(ref_file),
    '--meta_file', str(meta_new),
    '--scale', str(scale_blender),
    '--metallic', str(metallic),
    '--roughness', str(roughness),
    '--r', str(colour[0]),
    '--g', str(colour[1]),
    '--b', str(colour[2]),
    '--ior', str(ior)
]

run_command(cmd)

ref_file = ref_file + '.png'

# Calculate Magnitudes for light curve -------------------------------------
cmd = [
    'python', 'calculate_magnitudes.py',
    '--track_num', str(track_num),
    '--regime', str(regime),
    '--ref_file', str(ref_file),
    '--topo_data_file', str(topo_data_file),
    '--lc_plot_file', str(lc_plot_file),
    '--lc_track_file', str(lc_track_file),
    '--frames_dir', str(frames_file_path),
    '--meta_file', str(meta_new),
    '--scale', str(scale_orekit),
    '--obj', str(obj_name),
    '--rho_tot', str(rho_tot),
    '--fps', str(fps)
]
run_command(cmd)

end_time = time.time()
print(f"Track {track_num} completed in {(end_time - start_time)/60.0} minutes.")

with open(meta_new, 'a') as f:
    f.write(f"Computation Time: {(end_time - start_time)/60.0} minutes.\n")
