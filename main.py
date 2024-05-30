'''
calls other files
1. retrieve_ST_TLE.py
2. blender_sim.py as a blender script
'''

import subprocess 
import os
import sys
import random
from math import pi

def run_command(cmd):
    try:
        # Run the command with a timeout
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e.stderr}")
    except subprocess.TimeoutExpired:
        print("Command timed out")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

def create_directory(dir_name):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print(f"Script directory: {script_dir}")
    
    # Construct the full path for the new directory
    dir_path = script_dir + dir_name

    try:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory {dir_path} created successfully.")
    except Exception as e:
        print(f"Error creating directory {dir_path}: {e}")

    return

# Constants
scale_blender = 1/100
scale_orekit = 1/100000
so_bank = './objects'


track_num = 1
while track_num < 2:

    # Randomze parameters
    random_num = 1
    tle_file = f"./TLEs/tle_info{random_num}.txt"
    obj_name = '/antenna_1.stl'
    obj_path = so_bank + obj_name
    cross_sect = 0.1  # Cross section of satellite in m^2
    mass = 10.0 # Mass of satellite in kg
    cd = 2.2 # Coefficient of drag
    cr = 1.2  # coefficient of reflectivity 
    roughness = 0.3
    metallic = 1

    # Randomize attitude regime and spin rates
    regime = random.choice(['STABLE', 'TUMBLING'])
    spin_x = 2*pi
    spin_y = 2*pi
    spin_z = 2*pi

    
    # Create directory for new track
    '''
    track_directory = f'/data/track_{track_num}/'
    frames_file_path = track_directory + 'frames/'
    create_directory(track_directory)
    create_directory(frames_file_path)
    '''
    # Paths to pass 
    track_directory = f'./data/track_{track_num}/'
    frames_file_path = track_directory + 'frames/'
    ref_file = track_directory + 'reference_image'
    positions_file = track_directory + "obj_positions.txt"
    meta_file = track_directory + "metadata.txt"
    topo_data_file = track_directory + "topodata.txt"
    lc_plot_file = track_directory + f"lc_plot_{track_num}.png"
    lc_track_file = track_directory + f"lc_track_{track_num}.txt"
    
    # Run orbit propagator
    '''   
    cmd = [
            'python', 'tle_propagator.py', 
            '--track_dir', str(track_directory),
            '--obj_name', str(obj_name),
            '--obj_path', str(obj_path),
            '--tle_file', str(tle_file),
            '--positions_file', str(positions_file),
            '--meta_file', str(meta_file),
            '--topo_data_file', str(topo_data_file),
            '--scale', str(scale_orekit),
            '--regime', str(regime),
            '--spin_x', str(spin_x),
            '--spin_y', str(spin_y),
            '--spin_z', str(spin_z),
            '--mass', str(mass), # Mass of satellite in kg
            '--cross_sect', str(cross_sect), # Cross section of satellite in m^2
            '--cd', str(cd), # Coefficient of drag
            '--cr', str(cr) # Coefficient of reflectivity
          ]
    run_command(cmd)
    

    #Run blender simulation with object import
    cmd = [
        'blender', '-P', 'blender_sim.py', '--',
        '--obj_path', str(obj_path),
        '--meta_file', str(meta_file),
        '--frames_dir', str(frames_file_path),
        '--positions_file', str(positions_file),
        '--scale', str(scale_blender),
        '--metallic', str(metallic),
        '--roughness', str(roughness),
        '--spin_x', str(spin_x),
        '--spin_y', str(spin_y),
        '--spin_z', str(spin_z)

    ]
    run_command(cmd)
    
    # Run blender simulation for reference object
    cmd = [
        'blender', '-P', 'blender_sim_ref.py', '--',
        '--ref_file', str(ref_file),
        '--meta_file', str(meta_file),
        '--scale', str(scale_blender),
        '--metallic', str(metallic),
        '--roughness', str(roughness),
    ]
    ref_file = ref_file + '.png'
    run_command(cmd)
    '''
    ref_file = ref_file + '.png'
    # Calculate Magnitudes for light curve
    cmd = [
        'python', 'calculate_magnitudes.py',
        '--ref_file', str(ref_file),
        '--topo_data_file', str(topo_data_file),
        '--lc_plot_file', str(lc_plot_file),
        '--lc_track_file', str(lc_track_file),
        '--frames_dir', str(frames_file_path),
        '--meta_file', str(meta_file),
        '--cr', str(cr),
        '--scale', str(scale_blender),

    ]
    run_command(cmd)


    sys.exit()
    track_num += 1