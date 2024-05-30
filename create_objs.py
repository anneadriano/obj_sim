import bpy, bmesh
from math import radians, pi, cos, sin
import math

#---------------------------------------------------------------------------------
#FUNCTIONS

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

    scene = bpy.context.scene

def create_cone(radius, height, n_vertices): # roundnesss is a value between 0 and 1 (0 is no roundness)
    # Create a new mesh
    mesh = bpy.data.meshes.new(name="Cone")

    # Create a new object
    cone_obj = bpy.data.objects.new(name="Cone", object_data=mesh)

    # Link the object to the scene
    scene = bpy.context.scene
    scene.collection.objects.link(cone_obj)

    # Create cone geometry
    vertices = []
    edges = []
    faces = []

    # Add vertices for the base
    for i in range(n_vertices):
        angle = 2 * pi * (i / n_vertices)
        x = radius * cos(angle)
        y = radius * sin(angle)
        vertices.append((x, y, 0))

    # Add vertex for the tip
    vertices.append((0, 0, height))

    # Add edges for the base
    for i in range(n_vertices):
        edges.append((i, (i + 1) % n_vertices))

    # Add edges from the tip to the base vertices
    for i in range(n_vertices):
        edges.append((i, n_vertices))

    # Add faces for the base
    # faces.append(list(range(n_vertices)))

    # Add faces for the sides
    for i in range(n_vertices):
        faces.append([(i + 1) % n_vertices, i, n_vertices])

    # Set mesh data
    mesh.from_pydata(vertices, edges, faces)
    mesh.update()
    me =  cone_obj.data
    
    # Select the cone object
    bpy.context.view_layer.objects.active = cone_obj
    cone_obj.select_set(True)

    # Switch to edit mode to perform edge subdivision
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(me)

    # Deselect all edges
    for edge in bm.edges:
        edge.select = False

    # Select edges from tip to base
    for edge in bm.edges:
        if edge.verts[0].co.z == height and edge.verts[1].co.z == 0:
            edge.select = True
        elif edge.verts[0].co.z == 0 and edge.verts[1].co.z == height:
            edge.select = True

    # Update the mesh with the selected edges
    bmesh.update_edit_mesh(me)

    # Subdivide selected edges with one cut
    bpy.ops.mesh.subdivide(number_cuts=10)

    # Switch back to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Enable proportional editing
    scene.tool_settings.use_proportional_edit = True

    # Switch back to edit mode to apply proportional editing
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(me)

    # Deselect all vertices
    for vert in bm.verts:
        vert.select = False

    # Select base vertices
    # selected_verts = []
    # for vert in bm.verts:
    #     if vert.co.z == 0:
    #         vert.select = True
    #         # x = vert.co.x
    #         # y = vert.co.y
            
    #         # vert.select = False
       
    # bpy.ops.transform.resize(value=(1.2, 1.2, 0),
    #                                 constraint_axis=(False, False, False),
    #                                 orient_matrix_type='GLOBAL',
    #                                 mirror=False,
    #                                 use_proportional_edit=True,
    #                                 use_proportional_connected=True,
    #                                 proportional_edit_falloff='SMOOTH',
    #                                 proportional_size=1)

    selected_verts = []
    for vert in bm.verts:
        if vert.co.z == height:
            vert.select = True
            break
            # x = vert.co.x
            # y = vert.co.y
            
            # vert.select = False

    translation = height*0.3
    bpy.ops.transform.translate(value=(0, 0, -translation),
                                    constraint_axis=(True, True, False),
                                    orient_matrix_type='GLOBAL',
                                    mirror=False,
                                    use_proportional_edit=True,
                                    use_proportional_connected=True,
                                    proportional_edit_falloff='SMOOTH',
                                    proportional_size=1)



    height_thresh = 0.65*vert.co.z

    # Identify vertices above the height threshold
    vertices_to_delete = [v for v in bm.verts if v.co.z > height_thresh]
    
    # Delete vertices and associated geometry
    bmesh.ops.delete(bm, geom=vertices_to_delete, context='VERTS')

    # Switch back to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Update the mesh
    # bm.to_mesh(mesh)
    mesh.update()

    # Get the active object
    active_object = bpy.context.active_object

    # Check if there is an active object
    if active_object is not None:
        print("The active object is:", active_object.name)
    else:
        print("There is no active object.")

    return active_object
    
def create_rect(length, width, thick):
    '''
    This function is used to create both panels and buses
    '''
    # Add a cube primitive
    bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0))

    active_object = bpy.context.active_object

    # Scale the cube to the desired dimensions
    active_object.scale = (length / 2, width / 2, thick / 2)

    return active_object

def create_cyl(radius, height):
    # Create a new mesh
    mesh = bpy.data.meshes.new(name="Cylinder")

    # Create a new object
    cyl_obj = bpy.data.objects.new(name="Cylinder", object_data=mesh)

    # Link the object to the scene
    scene = bpy.context.scene
    scene.collection.objects.link(cyl_obj)

    # Create cylinder geometry
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, 
                                        radius=radius, 
                                        depth=height, 
                                        end_fill_type='NGON', 
                                        calc_uvs=True, 
                                        enter_editmode=False, 
                                        align='WORLD', 
                                        location=(0, 0, 0), 
                                        rotation=(0, 0, 0))

    active_object = bpy.context.active_object

    return active_object

def create_antenna(antenna_radius, sphere_radius):
    '''
    creates A large UV and keeps only the top
    '''
    cut_height = math.sqrt(sphere_radius**2 - (sphere_radius - antenna_radius)**2)

    # Create the UV sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius=sphere_radius, location=(0, 0, -cut_height))

    # Get a reference to the newly created sphere object
    object = bpy.context.active_object
    
    # Get the mesh data of the sphere and save to ne mesh
    mesh = object.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    # Vertices to delete
    vertices_to_delete = [vert for vert in bm.verts if vert.co.z < cut_height]

    # Delete the vertices along with associated edges and faces
    bmesh.ops.delete(bm, geom=vertices_to_delete, context='VERTS')
    
    bm.to_mesh(mesh)
    bm.free()

    


    return object


#---------------------------------------------------------------------------------
    
if __name__ == "__main__":

    # PARAMETERS ---------------------------------------------------------------
    save_path = './objects/'
    obj_name = 'panel_3.stl' # <--- change this for every object
    # Scales
    cone_scale = 0.75 # [0.75, 1.25]
    panel_scale_l = 6 # [1.0, 6.0]
    panel_scale_w = 3 # [1.0, 3.0]
    bus_scale_l = 14 # [1.0, 14.0]
    bus_scale_w = 14 # [1.0, 14.0]
    bus_scale_h = 37 # [1.0, 37.0]
    rocket_scale_h = 1.25 # [0.75, 1.25]
    rocket_rh_ratio = 0.134 # [0.1, 0.2]
    rod_scale_h = 1.5 # [0.5, 1.5]
    rod_rh_ratio = 0.01 # [0.005, 0.015]
    antenna_r_scale = 1 #[1, 3]
    # Cone parameters
    cone_r, cone_h, cone_vertices = 0.15, 1, 50
    # Panel parameters
    panel_l, panel_w, panel_t = 0.6, 0.62, 0.015
    # Bus parameters
    bus_l, bus_w, bus_h = 0.1, 0.1, 0.1
    # rocket parameters
    rocket_h = 13.8
    # rod parameters
    rod_h = 3
    # antenna parameters
    antenna_r = 0.5
    #------------------------------------------------------------------------------
    
    initialize_env()

    # Change function call here
    #active_obj = create_cone(cone_r*cone_scale, cone_h*cone_scale, cone_vertices)
    # active_obj = create_rect(panel_l*panel_scale_l, panel_w*panel_scale_w, panel_t) #panel version
    # active_obj = create_rect(bus_l*bus_scale_l, bus_w*bus_scale_w, bus_h*bus_scale_h) #bus version
    # active_obj = create_cyl(rocket_h*rocket_scale_h*rocket_rh_ratio, rocket_h*rocket_scale_h) # rocket version
    # active_obj = create_cyl(rod_h*rod_scale_h*rod_rh_ratio, rod_h*rod_scale_h) # rod version
    active_obj = create_antenna(antenna_r*antenna_r_scale, antenna_r_scale)


    # print("Saving active object to", save_path)
    # bpy.ops.export_mesh.stl(filepath=save_path+obj_name, 
    #                         check_existing=False, filter_glob="*.stl", 
    #                         use_selection=True, 
    #                         global_scale=1.0, 
    #                         use_scene_unit=False, 
    #                         ascii=False, 
    #                         use_mesh_modifiers=True, 
    #                         batch_mode='OFF', 
    #                         axis_forward='Y', 
    #                         axis_up='Z')


