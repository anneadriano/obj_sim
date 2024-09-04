'''
File to test importing pre-existing object .stl files
'''
import bpy, bmesh
from math import radians, pi, cos, sin
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
    # Set render engine to Cycles
    bpy.context.scene.render.engine = 'CYCLES'

    # Set the scene unit system to "Metric"
    bpy.context.scene.unit_settings.system = 'METRIC'

    # Set the scene length units to "Meters"
    bpy.context.scene.unit_settings.length_unit = 'METERS'

    # Clear existing mesh objects in the scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete()
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()

    enable_ambient_occlusion(scene, distance=1.0)

def setup_sun():
    bpy.ops.object.light_add(type='SUN', location=(200, -100, 5))
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

def set_surf_params(roughness, metallic, colour, ior):
    target = bpy.context.selected_objects[0]
    target.location = (0, 0, -0.3)
    # target.scale = (scale, scale, scale)
    target.hide_viewport = False
    target.hide_render = False
    target.name = 'target'

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


    #Following code in this function only for cone nad antenna

    # Switch to object mode if not already
    bpy.ops.object.mode_set(mode='OBJECT')

    # Apply smooth shading
    bpy.ops.object.shade_smooth()

    # Add a subdivision surface modifier
    subdiv = target.modifiers.new(name="Subdivision", type='SUBSURF')
    subdiv.levels = 3  # Viewport level
    subdiv.render_levels = 3 # Render level

def setup_camera():
    # bpy.ops.object.camera_add(location=(1, -0.7, 0.7), rotation=(0,0,0))
    bpy.ops.object.camera_add(location=(10, -4, 4), rotation=(0,0,0))

    # Select the camera object
    camera = bpy.context.active_object
    bpy.context.view_layer.objects.active = camera
    camera.select_set(True)

    # Set the camera name
    camera.name = "MyCamera"  # You can change "MyCamera" to your desired name
    
    constraint = camera.constraints.new(type='TRACK_TO') #camera stays pointed at the target
    constraint.target = bpy.data.objects["target"]  

    return camera

path = './objects/'
file_name = 'rod_3.stl'
image_loc = '/home/anne/Desktop/figures/'
colour = (0.6, 0.6, 0.6, 1.0)
roughness = 0.05
metallic = 0.8
ior = 2.0

initialize_env()
setup_sun()

bpy.ops.import_mesh.stl(filepath=path+file_name)
set_surf_params(roughness, metallic, colour, ior)
camera = setup_camera()

# #Render
# scene.camera = camera
# scene.render.image_settings.file_format = 'PNG'
# scene.render.resolution_x = 1920
# scene.render.resolution_y = 1080
# scene.render.filepath = image_loc + "specular_antenna"
# bpy.ops.render.render(write_still=True)



# Material properties for different surface regimes
'''
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
'''