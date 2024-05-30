import bpy
import math
import argparse
import sys
from math import radians, pi
import mathutils
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

def initialize_env():
    scene = bpy.context.scene
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

    # Set frames per second
    scene.render.fps = 6

    # Enable Ambient Occlusion - self-shadowing effect
    enable_ambient_occlusion(scene, distance=1.0)

    bpy.context.view_layer.update()

    return

def read_metadata(meta_file):

    with open(meta_file, "r") as file:
        # Read the lines of the file and create a list
        meta_data = file.readlines()

    cam_pos = meta_data[9].split(': ')[1].strip()
    sun_pos = meta_data[10].split(': ')[1].strip()
    zenith = meta_data[11].split(': ')[1].strip()
    ref_pos = meta_data[12].split(': ')[1].strip()

    print(ref_pos)
    

    return ref_pos, sun_pos, cam_pos, zenith

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

def create_ref_obj(ref_pos, roughness, metallic, zenith):
    
    x = float(ref_pos.split(',')[0][1:])
    y = float(ref_pos.split(',')[1]) 
    z = float(ref_pos.split(',')[2][:-1])
    x_zenith = float(zenith.split(';')[0][1:])
    y_zenith = float(zenith.split(';')[1])
    z_zenith = float(zenith.split(';')[2][:-1])
    
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z)) #placing object at a scaled altitude of 2000km
    ref = bpy.context.active_object
    ref.dimensions = (1, 1, 1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    ref.hide_viewport = False
    ref.hide_render = False
    # target.select_set(True)
    ref.name = 'target'

    # Create a principled shader for the cube (basic setup)
    mat = bpy.data.materials.new(name="targetMaterial")
    ref.data.materials.append(mat)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    principled_shader = nodes.get("Principled BSDF")
    if principled_shader:
        # principled_shader.location = (0, 0)
        # Set the roughness and metallic values directly on the Principled BSDF node
        principled_shader.inputs["Roughness"].default_value = roughness
        principled_shader.inputs["Metallic"].default_value = metallic
    
    # Rotate object
    target_normal = -mathutils.Vector((x_zenith, y_zenith, z_zenith))
    initial_normal = mathutils.Vector((0.0, 0.0, 1.0))
    # Calculate the rotation axis (cross product of initial and target normal)
    rotation_axis = initial_normal.cross(target_normal)
    if rotation_axis.length == 0:
        # If the cross product is zero, the vectors are collinear.
        # In this case, we rotate 180 degrees around any perpendicular axis.
        rotation_axis = mathutils.Vector((0, 1, 0))

    # Calculate the angle between the initial and target normal (dot product)
    angle = initial_normal.angle(target_normal)

    # Create the rotation matrix
    rotation_matrix = mathutils.Matrix.Rotation(angle, 4, rotation_axis)

    # Apply the rotation to the cube
    ref.matrix_world @= rotation_matrix
        
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

    return

def render(dir):
    # Set the render output format and file extension
    scene.render.image_settings.file_format = 'PNG'

    # Set the render resolution and frame range
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080

    # Set motion blur settings
    scene.camera = bpy.context.object
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 1.0  # Adjust shutter speed (1.0 is the default)

    # Render the animation
    bpy.context.scene.render.filepath = dir
    bpy.ops.render.render(write_still=True)

    print('DONE')

    return

# Argument Parser -------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Process parameters for Blender simulation.')
    parser.add_argument('--track_num', required=True, help='Track number')
    parser.add_argument('--ref_file', required=True, help='Directory to save tracking data')
    parser.add_argument('--meta_file', required=True, help='Path to the metadata file')
    parser.add_argument('--scale', required=True, type=float, help='Scale factor for blender distances')
    parser.add_argument('--metallic', required=True, type=float, help='Metallic property of the material')
    parser.add_argument('--roughness', required=True, type=float, help='Roughness of the material')

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    return args

# Parameters ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    
    # Save all arguments as variables
    track_num = args.track_num
    ref_file = args.ref_file
    meta_file = args.meta_file
    scale = args.scale
    metallic = args.metallic
    roughness = args.roughness

    # Constants ------------------------------------------------------------------------------------------
    Re = 6378*scale  # 6378 in km 
    focal_length = 5 #mm

    # Import positions and angles
    ref_pos, sun_pos, cam_pos, zenith = read_metadata(meta_file)

    # Blender environment initialization
    initialize_env()
    create_ref_obj(ref_pos, roughness, metallic, zenith)
    setup_sun(sun_pos, scale)
    setup_earth(Re)

    # Set up the camera
    setup_camera(focal_length, cam_pos, zenith)

    # Render and save key frames
    render(ref_file)

    # Finish
    print(f'*** Reference Image Generated for Track {track_num}. ***')
    bpy.ops.wm.quit_blender()  # Exit Blender after rendering
