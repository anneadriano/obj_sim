'''
File to test importing pre-existing object .stl files
'''
import bpy, bmesh
from math import radians, pi, cos, sin

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
    # bpy.ops.object.delete()
    # bpy.ops.object.select_by_type(type='CAMERA')
    # bpy.ops.object.delete()
    # bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()

    scene = bpy.context.scene

path = './objects/'
file_name = 'cone_6.stl'

initialize_env()
bpy.ops.import_mesh.stl(filepath=path+file_name)
