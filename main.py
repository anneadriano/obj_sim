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
import time

class object_params:
    def __init__(self, regime, shape):
        self.regime = regime
        self.shape = shape

    def update_properties(self): 
        antennas = ['antenna_1.stl', 'antenna_2.stl', 'antenna_3.stl']
        busses = ['bus_1.stl', 'bus_2.stl', 'bus_3.stl']
        cones = ['cone_1.stl', 'cone_2.stl', 'cone_3.stl']
        panels = ['panel_1.stl', 'panel_2.stl', 'panel_3.stl']
        rods = ['rod_1.stl', 'rod_2.stl', 'rod_3.stl']  

        # Shape properties
        # Conditions for reflector shape
        if self.shape in antennas:
            self.material = 'SPECULAR'
            self.mass = random.uniform(5.0, 10.0)
            self.cd = random.uniform(0.01,0.25)
            if self.shape == antennas[0]:
                self.cross_sect = 0.5
            elif self.shape == antennas[1]:
                self.cross_sect = 1.0
            else:
                self.cross_sect = 1.5

        # Conditions for panel shape
        elif self.shape in panels:
            self.material = random.choice(['SPECULAR', 'DIFFUSE', 'SOLAR'])
            self.mass = random.uniform(1.0, 5.0)
            self.cd = random.uniform(0.01,0.25)
            if self.shape == panels[0]:
                self.cross_sect = 0.3
            elif self.shape == panels[1]:
                self.cross_sect = 1.5
            else:
                self.cross_sect = 6.5

        #Other shape categories (bus, rod, cone)
        else:
            self.material = 'DIFFUSE'
            if self.shape in busses:
                self.cd = 0.03
                if self.shape == busses[0]:
                    self.cross_sect = 0.01
                    self.mass = 1.5
                elif self.shape == busses[1]:
                    self.cross_sect = 1.3
                    self.mass = 100.0
                elif self.shape == busses[2]:
                    self.cross_sect = 5.2
                    self.mass = 1000.0
            else:
                self.mass = random.uniform(0.5, 10.0)
                self.cd = 0.08
                if self.shape == cones[0]:
                    self.cross_sect = 0.1
                elif self.shape == cones[1]:
                    self.cross_sect = 0.15
                elif self.shape == cones[2]:
                    self.cross_sect = 0.23
                elif self.shape == rods[0]:
                    self.cross_sect = 0.0003125
                elif self.shape == rods[1]:
                    self.cross_sect = 0.0025
                else:
                    self.cross_sect = 0.02

        # Material properties
        if self.material == 'SPECULAR':
            self.colour = [0.6, 0.6, 0.6]
            self.roughness = 0.05
            self.metallic = 0.8
            self.rho_tot = 0.1
            self.ior = 2.0
            self.cr = 1.7
        elif self.material == 'DIFFUSE':
            self.colour = [0.6, 0.6, 0.6]
            self.roughness = 0.4
            self.metallic = 1.0
            self.rho_tot = 0.6
            self.ior = 1.5
            self.cr = 1.2
        else: # SOLAR (only panels)
            self.colour = [0.002, 0.001, 0.012]
            self.roughness = 0.1
            self.metallic = 0.8
            self.rho_tot = 0.6
            self.ior = 1.5
            self.cr = 0.5
        
    def update_attitude(self):
        min_spin = 2*pi/30 # 30 seconds per revolution
        max_spin = 2*pi/2 # 2 seconds per revolution
        if so.regime == 'STABLE':
            n_axes = random.choice([0,1])
            if n_axes == 0:
                spinx = 0.0
                spiny = 0.0
                spinz = 0.0
            else:
                axis = random.choice(['x', 'y', 'z'])
                if axis == 'x':
                    spinx = random.uniform(min_spin, max_spin)
                    spiny = 0.0
                    spinz = 0.0
                elif axis == 'y':
                    spinx = 0.0
                    spiny = random.uniform(min_spin, max_spin)
                    spinz = 0.0
                else:
                    spinx = 0.0
                    spiny = 0.0
                    spinz = random.uniform(min_spin, max_spin)
        else:
            n_axes = random.choice([2,3])
            if n_axes == 2:
                axes = random.choice(['xy', 'xz', 'yz'])
                if axes == 'xy':
                    spinx = random.uniform(min_spin, max_spin)
                    spiny = random.uniform(min_spin, max_spin)
                    spinz = 0.0
                elif axes == 'xz':
                    spinx = random.uniform(min_spin, max_spin)
                    spiny = 0.0
                    spinz = random.uniform(min_spin, max_spin)
                else:
                    spinx = 0.0
                    spiny = random.uniform(min_spin, max_spin)
                    spinz = random.uniform(min_spin, max_spin)
            else:
                spinx = random.uniform(min_spin, max_spin)
                spiny = random.uniform(min_spin, max_spin)
                spinz = random.uniform(min_spin, max_spin)
        
        self.spin = [spinx, spiny, spinz]

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

def randomize_object(so_bank):
    objects = os.listdir(so_bank)
    obj_name = random.choice(objects)
    obj_path = so_bank + obj_name

    return obj_name, obj_path

def randomize_tle():
    tle_num = random.randint(1, 10)
    tle_file = f"./TLEs/tle_info{tle_num}.txt"
    # tle_file = f"./TLEs/tle_info{8}.txt"

    return tle_file

# Constants
scale_blender = 1/100
scale_orekit = 1/100000
so_bank = './objects/'
n_frames = 400 # Number of frames to render
timestep = 0.2 # Timestep in seconds for orbit propagation
fps = 5 # Frames per second
# random.seed(42)

track_num = 1501
while track_num < 2001:
    
    # Create directory for new track
    obj_name, obj_path = randomize_object(so_bank)
    regime = random.choice(['STABLE', 'TUMBLING'])
    track_directory = f'/home/anne/Desktop/new/track_files/track_{track_num}/'
    frames_file_path = f'/home/anne/Desktop/new/frames/frames_{track_num}/'

    print(f"Starting track {track_num} with object {obj_name}.")

    create_directory(track_directory)
    create_directory(frames_file_path)
    
    # Paths to pass 
    ref_file = track_directory + 'reference_image'
    positions_file = track_directory + "obj_positions.txt"
    full_positions_file = track_directory + "full_obj_positions.txt"
    meta_file = track_directory + "metadata.txt"
    topo_data_file = track_directory + "topodata.txt"
    lc_plot_file = track_directory + f"lc_plot_{track_num}.png"
    lc_track_file = track_directory + f"lc_track_{track_num}.txt"
    lc_plot_noise_file = track_directory + f"lc_plot_noise_{track_num}.png"
    
    
    code = 1
    while code != 0:
        
        start_time = time.time()
        
        #Proper Randomization
        tle_file = randomize_tle()
        so = object_params(regime, obj_name)
        so.update_properties()
        so.update_attitude()

        # Randomize parameters ----------------------------------------------------
        # random_num = 4
        # tle_file = f"./TLEs/tle_info{random_num}.txt"
        # obj_name = 'rod_2.stl'
        # obj_path = so_bank + obj_name
        # so.cross_sect = 0.09  # Cross section of object in m^2
        # so.mass = 10.0 # Mass of object in kg
        # so.cd = 0.08 # Coefficient of drag
        # so.cr = 1.2  # coefficient of reflectivity 
        # so.roughness = 0.4
        # so.metallic = 1.0
        # so.rho_tot = 0.6 #Polished metallic surfaces have a total diffuse reflectance < total specular reflectance

        
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
                '--meta_file', str(meta_file),
                '--topo_data_file', str(topo_data_file),
                '--scale', str(scale_orekit),
                '--regime', str(so.regime),
                '--spin_x', str(so.spin[0]),
                '--spin_y', str(so.spin[1]),
                '--spin_z', str(so.spin[2]),
                '--mass', str(so.mass), # Mass of satellite in kg
                '--cross_sect', str(so.cross_sect), # Cross section of satellite in m^2
                '--cd', str(so.cd), # Coefficient of drag
                '--cr', str(so.cr), # Coefficient of reflectivity
                '--timestep', str(timestep), # Timestep in seconds
                '--num_frames', str(n_frames)
            ]
        code = run_command(cmd)
        print(f'Code: {code}')

    
    #Run blender simulation with object import --------------------------------
    # Start time
    
    
    cmd = [
        'blender', '-P', 'blender_sim.py', '--',
        '--track_num', str(track_num),
        '--obj_path', str(obj_path),
        '--meta_file', str(meta_file),
        '--frames_dir', str(frames_file_path),
        '--positions_file', str(positions_file),
        '--scale', str(scale_blender),
        '--metallic', str(so.metallic),
        '--roughness', str(so.roughness),
        '--spin_x', str(so.spin[0]),
        '--spin_y', str(so.spin[1]),
        '--spin_z', str(so.spin[2]),
        '--n_frames', str(n_frames),
        '--fps', str(fps),
        '--material', str(so.material),
        '--r', str(so.colour[0]),
        '--g', str(so.colour[1]),
        '--b', str(so.colour[2]),
        '--ior', str(so.ior)
    ]
    run_command(cmd)
    
    # Run blender simulation for reference object -----------------------------
    cmd = [
        'blender', '-P', 'blender_sim_ref.py', '--',
        '--track_num', str(track_num),
        '--ref_file', str(ref_file),
        '--meta_file', str(meta_file),
        '--scale', str(scale_blender),
        '--metallic', str(so.metallic),
        '--roughness', str(so.roughness),
        '--r', str(so.colour[0]),
        '--g', str(so.colour[1]),
        '--b', str(so.colour[2]),
        '--ior', str(so.ior)
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
        '--meta_file', str(meta_file),
        '--scale', str(scale_orekit),
        '--obj', str(obj_name),
        '--rho_tot', str(so.rho_tot),
        '--fps', str(fps)
    ]
    run_command(cmd)
    
    end_time = time.time()
    print(f"Track {track_num} completed in {(end_time - start_time)/60.0} minutes.")

    with open(meta_file, 'a') as f:
        f.write(f"Computation Time: {(end_time - start_time)/60.0} minutes.\n")
    
    track_num += 1