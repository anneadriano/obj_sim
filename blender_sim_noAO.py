import bpy
import math
import argparse
import sys
from math import radians, pi
import mathutils
import random
import os
scene = bpy.context.scene

def enable_ambient_occlusion(scene, distance=1.0):
    if scene.render.engine == 'CYCLES':
        # Add an Ambient Occlusion node to the World node tree
        world = scene.world
        world.use_nodes = True
        node_tree = world.node_tree
        nodes = node_tree.nodes
        
        # Clear existing nodes
        nodes.clear()
        
        # Add a background node
        background = nodes.new(type='ShaderNodeBackground')
        background.inputs['Color'].default_value = (0.02, 0.02, 0.02, 1)  # Dark gray background
        
        # Add an ambient occlusion node
        ao_node = nodes.new(type='ShaderNodeAmbientOcclusion')
        ao_node.inputs['Distance'].default_value = distance
        
        # Add an output node
        output = nodes.new(type='ShaderNodeOutputWorld')
        
        # Add a background node
        background = nodes.new(type='ShaderNodeBackground')
        background.inputs['Color'].default_value = (0, 0, 0, 1)  # Black background

        # Add an output node
        output = nodes.new(type='ShaderNodeOutputWorld')

        # Link nodes
        links = node_tree.links
        links.new(background.outputs['Background'], output.inputs['Surface'])
    
    elif scene.render.engine == 'BLENDER_EEVEE':
        scene.eevee.use_gtao = False  # Disable Ambient Occlusion for Eevee


    return

def initialize_env(fps):
    
    # Set render engine to Cycles
    scene.render.engine = 'CYCLES'

    # Set the scene unit system to "Metric"
    scene.unit_settings.system = 'METRIC'

    # Set the scene length units to "Meters"
    scene.unit_settings.length_unit = 'METERS'

    # Clear existing mesh objects in the scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete()
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()

    # Set the number of frames per second
    # 5 fps means 0.2s sampling frequency
    # 10 fps means 0.1s sampling frequency
    scene.render.fps = fps

    # Enable Ambient Occlusion - self-shadowing effect
    # enable_ambient_occlusion(scene, distance=1.0)

    bpy.context.view_layer.update()

    return

def read_positions(positions_path, meta_file):
    with open(positions_path, "r") as file:
        # Read the lines of the file and create a list
        list = file.readlines()
    # Remove newline characters from each list item
    obj_pos_list = [item.strip() for item in list]

    with open(meta_file, "r") as file:
        # Read the lines of the file and create a list
        meta_data = file.readlines()
    
    cam_pos = meta_data[9].split(': ')[1].strip()
    sun_pos = meta_data[10].split(': ')[1].strip()
    zenith = meta_data[11].split(': ')[1].strip()
    # print('Zenith: ', zenith)
    # print('Sun: ', sun_pos)
    # print('Cam: ', cam_pos)

    return obj_pos_list, sun_pos, cam_pos, zenith

def setup_sun(pos, scale):

    x = float(pos.split(',')[0][1:])
    y = float(pos.split(',')[1])
    z = float(pos.split(',')[2][:-1])

    bpy.ops.object.light_add(type='SUN', radius=696340*scale, align='WORLD', location=(x, y, z))#, location=(x, y, z))

    # Select the sun lamp
    sun_lamp = bpy.context.active_object

    # Set the strength of the sun lamp to 1062
    sun_lamp.data.energy = 1062.0

    # Set the background color to black
    scene.render.film_transparent = False  # Disable transparent background
    scene.world.color = (0, 0, 0)  # Set background color to black (RGB values)

    # Add blackbody shader to the world background
    sun_lamp.data.use_nodes = True 
    node_tree = sun_lamp.data.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    # Default emission node
    emit_node = nodes.get("Emission")
    if emit_node:
        # Add blackbody node
        blackbody_node = nodes.new(type="ShaderNodeBlackbody")
        blackbody_node.location = (emit_node.location.x, emit_node.location.y)
        blackbody_node.inputs['Temperature'].default_value = 5778
        links.new(blackbody_node.outputs['Color'], emit_node.inputs['Color'])

    bpy.context.view_layer.update()

    return

def setup_earth(Re):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=Re, location=(0, 0, 0))

    # Select the sphere object
    earth = bpy.context.active_object
    bpy.context.view_layer.objects.active = earth
    earth.select_set(True)

    # Rename the sphere
    earth.name = "earth"

    # Select the Earth object
    bpy.context.view_layer.objects.active = bpy.data.objects['earth']

    # Check if the Earth object has a material
    if not earth.data.materials:
        # If no material, create a new one
        earth.data.materials.append(bpy.data.materials.new(name="EarthMaterial"))

    # Get the active material
    earth_material = earth.active_material

    # Ensure that the material uses the Principled BSDF shader
    if earth_material.node_tree is None:
        earth_material.use_nodes = True
        earth_material.node_tree.links.clear()  # Clear default nodes

    # Set the albedo color (Base Color) using RGB values
    albedo_color = (0.3, 0.3, 0.3, 1.0)
    earth_material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = albedo_color

    bpy.context.view_layer.update()


    return

def randomize_track_start(pos_list, num_frames):
    int_range = range(0,len(pos_list)-num_frames)
    start_index = random.choice(int_range)

    return start_index, len(pos_list)

def import_obj(path, pos_list, roughness, metallic, num_frames, spin_state, meta_file, ior, colour, material): 
    bpy.ops.import_mesh.stl(filepath=path) 
    # bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
    # target = bpy.context.active_object
    # bpy.context.view_layer.objects.active = target
    target = bpy.context.selected_objects[0]
    # target.scale = (scale, scale, scale)
    target.hide_viewport = False
    target.hide_render = False
    target.name = 'target'


    #Randomize rotation
    random_rotation = (random.uniform(0, 2 * math.pi),  # Random rotation around X axis
                       random.uniform(0, 2 * math.pi),  # Random rotation around Y axis
                       random.uniform(0, 2 * math.pi)   # Random rotation around Z axis
                      )

    target.rotation_euler = random_rotation


    # Create a principled shader for the cube (basic setup)
    mat = bpy.data.materials.new(name="targetMaterial")
    target.data.materials.append(mat)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    principled_shader = nodes.get("Principled BSDF")
    if principled_shader:
        # principled_shader.location = (0, 0)
        # Set the roughness and metallic values directly on the Principled BSDF node
        principled_shader.inputs["Roughness"].default_value = roughness
        principled_shader.inputs["Metallic"].default_value = metallic
        principled_shader.inputs['Base Color'].default_value = colour
        principled_shader.inputs["IOR"].default_value = ior
    
    #Subdivide surface for smoother finish
    if 'antenna' in path or  'cone' in path:
        # Switch to object mode if not already
        bpy.ops.object.mode_set(mode='OBJECT')

        # Apply smooth shading
        bpy.ops.object.shade_smooth()

        # Add a subdivision surface modifier
        subdiv = target.modifiers.new(name="Subdivision", type='SUBSURF')
        subdiv.levels = 3  # Viewport level
        subdiv.render_levels = 3 # Render level

    # Calculate the delta rotation angle for each frame
    rot_angle_x = spin_state[0] / scene.render.fps
    rot_angle_y = spin_state[1] / scene.render.fps
    rot_angle_z = spin_state[2] / scene.render.fps

    # Set start index
    start_index, total_vis = randomize_track_start(pos_list, num_frames)
    
    # Write start index of position list to metadata file
    with open(meta_file, "a") as file:
        file.write(f'Material: {material}\n')
        file.write(f'Roughness: {roughness}\n')
        file.write(f'Metallic: {metallic}\n')
        file.write(f'Index of Refraction: {ior}\n')
        file.write(f'Base Colour: r: {colour[0]}, g: {colour[1]}, b: {colour[2]}\n')
        file.write(f"Position Start Index: {start_index}\n")
        file.write(f"Total Visible Positions: {total_vis}\n")
        file.write(f'Initial Orientation: {random_rotation}\n')

    # Create keyframes for rotation
    for frame in range(1, num_frames + 1):
        scene.frame_set(frame)
        target.rotation_euler = (rot_angle_x * frame, rot_angle_y * frame, rot_angle_z * frame)  # Adjust the rotation axis as needed
        target.keyframe_insert(data_path="rotation_euler", frame = frame)

        coord = pos_list[start_index + frame-1]
        x = float(coord.split(',')[0][1:])
        y = float(coord.split(',')[1])
        z = float(coord.split(',')[2][:-1])
        target.location = (x, y, z)
        target.keyframe_insert(data_path="location", frame=frame)
    
    bpy.context.view_layer.update()

    return

def setup_camera(focal_length, cam_pos, zenith):
    x = float(cam_pos.split(',')[0][1:])
    y = float(cam_pos.split(',')[1]) 
    z = float(cam_pos.split(',')[2][:-1])
    bpy.ops.object.camera_add(location=(x, y, z), rotation=(0,0,0))

    # Select the camera object
    camera = bpy.context.active_object
    bpy.context.view_layer.objects.active = camera
    camera.select_set(True)

    # Set the camera name
    camera.name = "MyCamera"  # You can change "MyCamera" to your desired name
    camera.data.lens = focal_length
    
    # Point the camera 
    x_zenith = float(zenith.split(';')[0][1:])
    y_zenith = float(zenith.split(';')[1])
    z_zenith = float(zenith.split(';')[2][:-1])

    local_z_axis = mathutils.Vector((0.0, 0.0, 1.0))
    zenith_vector = mathutils.Vector((x_zenith, y_zenith, z_zenith))

    # Compute the rotation quaternion
    rotation_quaternion = local_z_axis.rotation_difference(zenith_vector)

    # Apply the rotation to the camera
    camera.rotation_mode = 'QUATERNION'
    camera.rotation_quaternion = rotation_quaternion
    
    bpy.context.view_layer.update()

    return camera

def render(dir, num_frames, meta_file):
    # Set the render output format and file extension
    scene.render.image_settings.file_format = 'PNG'

    # Set the render resolution and frame range
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080


    # Set motion blur settings
    scene.camera = bpy.context.object
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 1.0  # Adjust shutter speed (1.0 is the default)

    not_saved = []

    for frame in range(1, num_frames + 1):

        try:
            #Get image of target
            scene.frame_set(frame)
            scene.render.filepath = dir + "frame_" + str(frame)
            bpy.ops.render.render(write_still=True)
        except Exception as e:
            print(f"Error rendering frame {frame}: {e}", file=sys.stderr)
            not_saved.append(frame)


    print('DONE')

    return not_saved

def point_and_crop(camera):
    constraint = camera.constraints.new(type='TRACK_TO') #camera stays pointed at the target
    constraint.target = bpy.data.objects["target"]  
    
    # Set the area to be rendered (normalized coordinates: 0 to 1)
    min_x = 0.4
    min_y = 0.4
    max_x = 0.6
    max_y = 0.6

    # Set the render border
    scene.render.border_min_x = min_x
    scene.render.border_min_y = min_y
    scene.render.border_max_x = max_x
    scene.render.border_max_y = max_y

    # Enable render border
    scene.render.use_border = True
    scene.render.use_crop_to_border = True

    return


# Argument Parser -------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Process parameters for Blender simulation.')
    parser.add_argument('--track_num', required=True, help='Track number')
    parser.add_argument('--obj_path', required=True, help='Path to the .stl file to be imported')
    parser.add_argument('--meta_file', required=True, help='Path to the metadata file')
    parser.add_argument('--frames_dir', required=True, help='Directory to save frame images')
    parser.add_argument('--positions_file', required=True, help='Path to the object positions file')
    parser.add_argument('--scale', required=True, type=float, help='Scale factor for object dimensions')
    parser.add_argument('--metallic', required=True, type=float, help='Metallic property of the material')
    parser.add_argument('--roughness', required=True, type=float, help='Roughness of the material')
    parser.add_argument('--spin_x', required=True, type=float, help='x-axis spin rate in rad/s')
    parser.add_argument('--spin_y', required=True, type=float, help='y-axis spin rate in rad/s')
    parser.add_argument('--spin_z', required=True, type=float, help='z-axis spin rate in rad/s')
    parser.add_argument('--n_frames', required=True, type=int, help='Number of frames to render')
    parser.add_argument('--fps', required=True, type=int, help='Frames per second')
    parser.add_argument('--material', required=True, help='Type of material')
    parser.add_argument('--ior', required=True, type=float, help='Index of refraction')
    parser.add_argument('--r', required=True, type=float, help='RGB value for the object color')
    parser.add_argument('--g', required=True, type=float, help='RGB value for the object color')
    parser.add_argument('--b', required=True, type=float, help='RGB value for the object color')

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    return args

# Parameters ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    
    # Save all arguments as variables
    track_num = args.track_num
    obj_path = args.obj_path
    meta_file = args.meta_file
    frames_dir = args.frames_dir
    positions_file = args.positions_file
    scale = args.scale
    metallic = args.metallic
    roughness = args.roughness
    spin_x = args.spin_x
    spin_y = args.spin_y
    spin_z = args.spin_z
    num_frames = args.n_frames
    fps = args.fps
    material = args.material
    ior = args.ior
    r = args.r
    g = args.g
    b = args.b
    colour = (r, g, b, 1.0)

    # Constants
    Re = 6378*scale  # 6378 in km 
    spin_state = (spin_x,spin_y,spin_z)  # (x, y, z) spin rates [radians/second]
    #Camera parameters
    # cam_long = radians(41.4)
    # cam_lat = radians(43.8)
    # H = (2030/1000)*scale #km scaled
    focal_length = 5 #mm`

    # Import positions and angles
    obj_pos_list, sun_pos, cam_pos, zenith = read_positions(positions_file, meta_file)
    scene.frame_end = num_frames
    # print('Positions: ', len(obj_pos_list))

    # Blender environment initialization
    print('initializing Blender environment...')
    initialize_env(fps)
    import_obj(obj_path, obj_pos_list, roughness, metallic, num_frames, spin_state, meta_file, ior, colour, material)
    setup_sun(sun_pos, scale)
    setup_earth(Re)

    # Set up the camera
    camera = setup_camera(focal_length, cam_pos, zenith)
    point_and_crop(camera)

    # Render and save key frames
    not_saved = render(frames_dir, num_frames, meta_file)

    with open(meta_file, "a") as file:
        file.write(f'Frames Not Saved ({len(not_saved)}): {not_saved}\n')

    # Finish
    print(f'*** Frame Images Generated for Track {track_num}. ***')
    
    os._exit(0)