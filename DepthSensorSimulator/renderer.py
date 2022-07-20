import os
import random
import bpy
import math
import numpy as np
from mathutils import Vector, Matrix
import copy
import sys
import time
from bpy_extras.object_utils import world_to_camera_view

sys.path.append(os.getcwd())
from modify_material import set_modify_material, set_modify_raw_material

argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"
RENDERING_PATH = os.getcwd()
SCENE_NUM = argv[0] if len(argv) > 0 else 0


###########################
# Parameter setting
###########################
working_root = "/data/sensor/renderer/DepthSensorSimulator"
CAD_model_root_path = os.path.join(working_root, "cad_model")                           # CAD model path
env_map_path = os.path.join(working_root, "envmap_lib")                                 # envirment map path
output_root_path = os.path.join(working_root, "rendered_output")                        # rendered output path
DEVICE_LIST = [0]                                                                       # GPU id

# rendered output setting (0: no output, 1: output)
render_mode_list = {'RGB': 1,
                    'IR': 1,
                    'NOCS': 1,
                    'Mask': 1,  
                    'Normal': 1}

# material randomization mode (transparent, specular, mixed, raw)
my_material_randomize_mode = 'mixed'

# set depth sensor parameter
camera_width = 1280
camera_height = 720
camera_fov = 71.28 / 180 * math.pi
baseline_distance = 0.055
num_frame_per_scene = 30    # number of cameras per scene
LIGHT_EMITTER_ENERGY = 5
LIGHT_ENV_MAP_ENERGY_IR = 0.035
LIGHT_ENV_MAP_ENERGY_RGB = 0.5

# set background parameter
background_size = 3.
background_position = (0., 0., 0.)
background_scale = (1., 1., 1.)

# set camera randomized paramater
# start_point_range: (range_r, range_vector),   range_r: (r_min, r_max),    range_vector: (x_min, x_max, y_min, y_max)
# look_at_range: (x_min, x_max, y_min, y_max, z_min, z_max)
# up_range: (x_min, x_max, y_min, y_max)
start_point_range = ((0.5, 0.95), (-0.6, 0.6, -0.6, 0.6))
up_range = (-0.18, -0.18, -0.18, 0.18)
look_at_range = (background_position[0] - 0.05, background_position[0] + 0.05, 
                 background_position[1] - 0.05, background_position[1] + 0.05,
                 background_position[2] - 0.05, background_position[2] + 0.05)


g_syn_light_num_lowbound = 4
g_syn_light_num_highbound = 6
g_syn_light_dist_lowbound = 8
g_syn_light_dist_highbound = 12
g_syn_light_azimuth_degree_lowbound = 0
g_syn_light_azimuth_degree_highbound = 360
g_syn_light_elevation_degree_lowbound = 0
g_syn_light_elevation_degree_highbound = 90
g_syn_light_energy_mean = 3
g_syn_light_energy_std = 0.5
g_syn_light_environment_energy_lowbound = 0
g_syn_light_environment_energy_highbound = 1


g_shape_synset_name_pairs_all = {'02691156': 'aeroplane',
                                '02747177': 'ashtray',
                                '02773838': 'backpack',
                                '02801938': 'basket',
                                '02808440': 'tub',  # bathtub
                                '02818832': 'bed',
                                '02828884': 'bench',
                                '02834778': 'bicycle',
                                '02843684': 'mailbox', # missing in objectnet3d, birdhouse, use view distribution of mailbox
                                '02858304': 'boat',
                                '02871439': 'bookshelf',
                                '02876657': 'bottle',
                                '02880940': 'bowl', # missing in objectnet3d, bowl, use view distribution of plate
                                '02924116': 'bus',
                                '02933112': 'cabinet',
                                '02942699': 'camera',
                                '02946921': 'can',
                                '02954340': 'cap',
                                '02958343': 'car',
                                '02992529': 'cellphone',
                                '03001627': 'chair',
                                '03046257': 'clock',
                                '03085013': 'keyboard',
                                '03207941': 'dishwasher',
                                '03211117': 'tvmonitor',
                                '03261776': 'headphone',
                                '03325088': 'faucet',
                                '03337140': 'filing_cabinet',
                                '03467517': 'guitar',
                                '03513137': 'helmet',
                                '03593526': 'jar',
                                '03624134': 'knife',
                                '03636649': 'lamp',
                                '03642806': 'laptop',
                                '03691459': 'speaker',
                                '03710193': 'mailbox',
                                '03759954': 'microphone',
                                '03761084': 'microwave',
                                '03790512': 'motorbike',
                                '03797390': 'mug',  # missing in objectnet3d, mug, use view distribution of cup
                                '03928116': 'piano',
                                '03938244': 'pillow',
                                '03948459': 'rifle',  # missing in objectnet3d, pistol, use view distribution of rifle
                                '03991062': 'pot',
                                '04004475': 'printer',
                                '04074963': 'remote_control',
                                '04090263': 'rifle',
                                '04099429': 'road_pole',  # missing in objectnet3d, rocket, use view distribution of road_pole
                                '04225987': 'skateboard',
                                '04256520': 'sofa',
                                '04330267': 'stove',
                                '04379243': 'diningtable',  # use view distribution of dining_table
                                '04401088': 'telephone',
                                '04460130': 'road_pole',  # missing in objectnet3d, tower, use view distribution of road_pole
                                '04468005': 'train',
                                '04530566': 'washing_machine',
                                '04554684': 'dishwasher'}  # washer, use view distribution of dishwasher


###########################
# Utils
###########################
def obj_centered_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)

def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)    
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)    
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)

def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist    
    t = math.sqrt(cx * cx + cy * cy) 
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx*cx + ty*cy, -1),1)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll    
    print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)    
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)

def camRotQuaternion(cx, cy, cz, theta): 
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)

def quaternionProduct(qx, qy): 
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e    
    return (q1, q2, q3, q4)

def quaternionToRotation(q):
    w, x, y, z = q
    r00 = 1 - 2 * y ** 2 - 2 * z ** 2
    r01 = 2 * x * y + 2 * w * z
    r02 = 2 * x * z - 2 * w * y

    r10 = 2 * x * y - 2 * w * z
    r11 = 1 - 2 * x ** 2 - 2 * z ** 2
    r12 = 2 * y * z + 2 * w * x

    r20 = 2 * x * z + 2 * w * y
    r21 = 2 * y * z - 2 * w * x
    r22 = 1 - 2 * x ** 2 - 2 * y ** 2
    r = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
    return r

def quaternionFromRotMat(rotation_matrix):
    rotation_matrix = np.reshape(rotation_matrix, (1, 9))[0]
    w = math.sqrt(rotation_matrix[0]+rotation_matrix[4]+rotation_matrix[8]+1 + 1e-6)/2
    x = math.sqrt(rotation_matrix[0]-rotation_matrix[4]-rotation_matrix[8]+1 + 1e-6)/2
    y = math.sqrt(-rotation_matrix[0]+rotation_matrix[4]-rotation_matrix[8]+1 + 1e-6)/2
    z = math.sqrt(-rotation_matrix[0]-rotation_matrix[4]+rotation_matrix[8]+1 + 1e-6)/2
    a = [w,x,y,z]
    m = a.index(max(a))
    if m == 0:
        x = (rotation_matrix[7]-rotation_matrix[5])/(4*w)
        y = (rotation_matrix[2]-rotation_matrix[6])/(4*w)
        z = (rotation_matrix[3]-rotation_matrix[1])/(4*w)
    if m == 1:
        w = (rotation_matrix[7]-rotation_matrix[5])/(4*x)
        y = (rotation_matrix[1]+rotation_matrix[3])/(4*x)
        z = (rotation_matrix[6]+rotation_matrix[2])/(4*x)
    if m == 2:
        w = (rotation_matrix[2]-rotation_matrix[6])/(4*y)
        x = (rotation_matrix[1]+rotation_matrix[3])/(4*y)
        z = (rotation_matrix[5]+rotation_matrix[7])/(4*y)
    if m == 3:
        w = (rotation_matrix[3]-rotation_matrix[1])/(4*z)
        x = (rotation_matrix[6]+rotation_matrix[2])/(4*z)
        y = (rotation_matrix[5]+rotation_matrix[7])/(4*z)
    quaternion = (w,x,y,z)
    return quaternion

def rotVector(q, vector_ori):
    r = quaternionToRotation(q)
    x_ori = vector_ori[0]
    y_ori = vector_ori[1]
    z_ori = vector_ori[2]
    x_rot = r[0][0] * x_ori + r[1][0] * y_ori + r[2][0] * z_ori
    y_rot = r[0][1] * x_ori + r[1][1] * y_ori + r[2][1] * z_ori
    z_rot = r[0][2] * x_ori + r[1][2] * y_ori + r[2][2] * z_ori
    return (x_rot, y_rot, z_rot)

def cameraLPosToCameraRPos(q_l, pos_l, baseline_dis):
    vector_camera_l_y = (1, 0, 0)
    vector_rot = rotVector(q_l, vector_camera_l_y)
    pos_r = (pos_l[0] + vector_rot[0] * baseline_dis,
             pos_l[1] + vector_rot[1] * baseline_dis,
             pos_l[2] + vector_rot[2] * baseline_dis)
    return pos_r

def getRTFromAToB(pointCloudA, pointCloudB):

    muA = np.mean(pointCloudA, axis=0)
    muB = np.mean(pointCloudB, axis=0)

    zeroMeanA = pointCloudA - muA
    zeroMeanB = pointCloudB - muB

    covMat = np.matmul(np.transpose(zeroMeanA), zeroMeanB)
    U, S, Vt = np.linalg.svd(covMat)
    R = np.matmul(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T * U.T
    T = (-np.matmul(R, muA.T) + muB.T).reshape(3, 1)
    return R, T

def cameraPositionRandomize(start_point_range, look_at_range, up_range):
    r_range, vector_range = start_point_range
    r_min, r_max = r_range
    x_min, x_max, y_min, y_max = vector_range
    r = random.uniform(r_min, r_max)
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)
    z = math.sqrt(1 - x**2 - y**2)
    vector_camera_axis = np.array([x, y, z])

    x_min, x_max, y_min, y_max = up_range
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)    
    z = math.sqrt(1 - x**2 - y**2)
    up = np.array([x, y, z])

    x_min, x_max, y_min, y_max, z_min, z_max = look_at_range
    look_at = np.array([random.uniform(x_min, x_max),
                        random.uniform(y_min, y_max),
                        random.uniform(z_min, z_max)])
    position = look_at + r * vector_camera_axis

    vectorZ = - (look_at - position)/np.linalg.norm(look_at - position)
    vectorX = np.cross(up, vectorZ)/np.linalg.norm(np.cross(up, vectorZ))
    vectorY = np.cross(vectorZ, vectorX)/np.linalg.norm(np.cross(vectorX, vectorZ))

    # points in camera coordinates
    pointSensor= np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])

    # points in world coordinates 
    pointWorld = np.array([position,
                            position + vectorX,
                            position + vectorY * 2,
                            position + vectorZ * 3])

    resR, resT = getRTFromAToB(pointSensor, pointWorld)
    resQ = quaternionFromRotMat(resR)
    return resQ, resT    

def quanternion_mul(q1, q2):
    s1 = q1[0]
    v1 = np.array(q1[1:])
    s2 = q2[0]
    v2 = np.array(q2[1:])
    s = s1 * s2 - np.dot(v1, v2)
    v = s1 * v2 + s2 * v1 + np.cross(v1, v2)
    return (s, v[0], v[1], v[2])

def setModelPosition(instance, position_limit, instance_mask_id):
    x_min, x_max, y_min, y_max, z = position_limit
    instance.rotation_mode = 'XYZ'
    instance.rotation_euler = (random.uniform(math.pi/2 - math.pi/4, math.pi/2 + math.pi/4), random.uniform(- math.pi/4, math.pi/4), random.uniform(-math.pi, math.pi))
    instance.location = (random.uniform(x_min, x_max), random.uniform(y_min, y_max), z + instance_mask_id * 0.1)

def setRigidBody(instance):
    bpy.context.view_layer.objects.active = instance 
    object_single = bpy.context.active_object

    # add rigid body constraints to cube
    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.mass = 1
    bpy.context.object.rigid_body.kinematic = True
    bpy.context.object.rigid_body.collision_shape = 'CONVEX_HULL'
    bpy.context.object.rigid_body.restitution = 0.01
    bpy.context.object.rigid_body.angular_damping = 0.8
    bpy.context.object.rigid_body.linear_damping = 0.99

    bpy.context.object.rigid_body.kinematic = False
    object_single.keyframe_insert(data_path='rigid_body.kinematic', frame=0)

def set_visiable_objects(visible_objects_list):
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and not obj.name.split('_')[0] == 'background':
            if obj in visible_objects_list:
                obj.hide_render = False
            else:
                obj.hide_render = True

def generate_CAD_model_list(model_path):
    CAD_model_list = {}
    for class_folder in os.listdir(model_path):
        if class_folder[0] == '.':
            continue
        class_path = os.path.join(model_path, class_folder)
        class_name = g_shape_synset_name_pairs[class_folder] if class_folder in g_shape_synset_name_pairs else 'other'
        class_list = []
        for instance_folder in os.listdir(class_path):
            if instance_folder[0] == '.':
                continue
            instance_path = os.path.join(class_path, instance_folder, "model.obj")
            class_list.append([instance_path, class_name])
        if class_name == 'other' and 'other' in CAD_model_list:
            CAD_model_list[class_name] = CAD_model_list[class_name] + class_list
        else:
            CAD_model_list[class_name] = class_list

    return CAD_model_list

def generate_material_type(obj_name):
    flag = random.randint(0, 3)
    # select the raw material
    if flag == 0:
        flag = random.randint(0, 1)
        if flag == 0:
            return 'raw'
        else:            
            if obj_name.split('_')[1] in class_material_pairs['transparent']:
                return 'diffuse'                     
    # select one from specular and transparent
    else:
        flag = random.randint(0, 2)
        if flag < 2:
            if obj_name.split('_')[1] in class_material_pairs['transparent']:
                return 'transparent'
            else:
                flag = 2

        if flag == 2:
            if obj_name.split('_')[1] in class_material_pairs['specular']:
                return 'specular'  
            else:
                return 'diffuse'
    return 'raw'


###########################
# Renderer Class
###########################
class BlenderRenderer(object):

    def __init__(self, viewport_size_x=640, viewport_size_y=360):
        '''
        viewport_size_x, viewport_size_y: rendering viewport resolution
        '''

        # remove all objects, cameras and lights
        for obj in bpy.data.meshes:
            bpy.data.meshes.remove(obj)

        for cam in bpy.data.cameras:
            bpy.data.cameras.remove(cam)

        for light in bpy.data.lights:
            bpy.data.lights.remove(light)

        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        # remove all materials
        # for item in bpy.data.materials:
        #     bpy.data.materials.remove(item)

        render_context = bpy.context.scene.render

        # add left camera
        camera_l_data = bpy.data.cameras.new(name="camera_l")
        camera_l_object = bpy.data.objects.new(name="camera_l", object_data=camera_l_data)
        bpy.context.collection.objects.link(camera_l_object)

        # add right camera
        camera_r_data = bpy.data.cameras.new(name="camera_r")
        camera_r_object = bpy.data.objects.new(name="camera_r", object_data=camera_r_data)
        bpy.context.collection.objects.link(camera_r_object)

        camera_l = bpy.data.objects["camera_l"]
        camera_r = bpy.data.objects["camera_r"]

        # set the camera postion and orientation so that it is in
        # the front of the object
        camera_l.location = (1, 0, 0)
        camera_r.location = (1, 0, 0)

        # add emitter light
        light_emitter_data = bpy.data.lights.new(name="light_emitter", type='SPOT')
        light_emitter_object = bpy.data.objects.new(name="light_emitter", object_data=light_emitter_data)
        bpy.context.collection.objects.link(light_emitter_object)

        light_emitter = bpy.data.objects["light_emitter"]
        light_emitter.location = (1, 0, 0)
        light_emitter.data.energy = LIGHT_EMITTER_ENERGY

        # render setting
        render_context.resolution_percentage = 100
        self.render_context = render_context

        self.camera_l = camera_l
        self.camera_r = camera_r

        self.light_emitter = light_emitter

        self.model_loaded = False
        self.background_added = None

        self.render_context.resolution_x = viewport_size_x
        self.render_context.resolution_y = viewport_size_y

        self.my_material = {}
        self.render_mode = 'IR'

        # output setting 
        self.render_context.image_settings.file_format = 'PNG'
        self.render_context.image_settings.compression = 0
        self.render_context.image_settings.color_mode = 'BW'
        self.render_context.image_settings.color_depth = '8'

        # cycles setting
        self.render_context.engine = 'CYCLES'
        bpy.context.scene.cycles.progressive = 'BRANCHED_PATH'
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = 'NLM'
        bpy.context.scene.cycles.film_exposure = 0.5

        # self.render_context.use_antialiasing = False
        bpy.context.scene.view_layers["View Layer"].use_sky = True

        # switch on nodes
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links
  
        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)
  
        # create input render layer node
        rl = tree.nodes.new('CompositorNodeRLayers')

        # create output node
        self.fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
        self.fileOutput.base_path = "./new_data/0000"
        self.fileOutput.format.file_format = 'OPEN_EXR'
        self.fileOutput.format.color_depth= '32'
        self.fileOutput.file_slots[0].path = 'depth#'
        # links.new(map.outputs[0], fileOutput.inputs[0])
        links.new(rl.outputs[2], self.fileOutput.inputs[0])
        # links.new(gamma.outputs[0], fileOutput.inputs[0])

        # depth sensor pattern
        self.pattern = []
        # environment map
        self.env_map = []


    def loadImages(self, env_map_path):
        for img in bpy.data.images:
            if img.filepath.split("/")[-1] == "pattern.png":
                self.pattern = img
                break
        for item in os.listdir(env_map_path):
            if item.split('.')[-1] == 'hdr':
                self.env_map.append(bpy.data.images.load(filepath=os.path.join(env_map_path, item)))


    def addEnvMap(self):
        # Get the environment node tree of the current scene
        node_tree = bpy.context.scene.world.node_tree
        tree_nodes = node_tree.nodes

        # Clear all nodes
        tree_nodes.clear()

        # Add Background node
        node_background = tree_nodes.new(type='ShaderNodeBackground')

        # Add Environment Texture node
        node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
        # Load and assign the image to the node property
        # node_environment.image = bpy.data.images.load("/Users/zhangjiyao/Desktop/test_addon/envmap_lib/autoshop_01_1k.hdr") # Relative path
        node_environment.location = -300,0

        node_tex_coord = tree_nodes.new(type='ShaderNodeTexCoord')
        node_tex_coord.location = -700,0

        node_mapping = tree_nodes.new(type='ShaderNodeMapping')
        node_mapping.location = -500,0

        # Add Output node
        node_output = tree_nodes.new(type='ShaderNodeOutputWorld')   
        node_output.location = 200,0

        # Link all nodes
        links = node_tree.links
        links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
        links.new(node_background.outputs["Background"], node_output.inputs["Surface"])
        links.new(node_tex_coord.outputs["Generated"], node_mapping.inputs["Vector"])
        links.new(node_mapping.outputs["Vector"], node_environment.inputs["Vector"])

        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 1.0


    def setEnvMap(self, env_map_id, rotation_elur_z):
        # Get the environment node tree of the current scene
        node_tree = bpy.context.scene.world.node_tree

        # Get Environment Texture node
        node_environment = node_tree.nodes['Environment Texture']
        # Load and assign the image to the node property
        node_environment.image = self.env_map[env_map_id]

        node_mapping = node_tree.nodes['Mapping']
        node_mapping.inputs[2].default_value[2] = rotation_elur_z


    def addMaskMaterial(self, num=20):
        material_name = "mask_background"

        # test if material exists
        # if it does not exist, create it:
        material_class = (bpy.data.materials.get(material_name) or 
            bpy.data.materials.new(material_name))

        # enable 'Use nodes'
        material_class.use_nodes = True
        node_tree = material_class.node_tree

        # remove default nodes
        material_class.node_tree.nodes.clear()

        # add new nodes  
        node_1 = node_tree.nodes.new('ShaderNodeOutputMaterial')
        node_2= node_tree.nodes.new('ShaderNodeBrightContrast')

        # link nodes
        node_tree.links.new(node_1.inputs[0], node_2.outputs[0])
        node_2.inputs[0].default_value = (1, 1, 1, 1)
        self.my_material[material_name] =  material_class

        for i in range(num):
            class_name = str(i + 1)
            # set the material of background    
            material_name = "mask_" + class_name

            # test if material exists
            # if it does not exist, create it:
            material_class = (bpy.data.materials.get(material_name) or 
                bpy.data.materials.new(material_name))

            # enable 'Use nodes'
            material_class.use_nodes = True
            node_tree = material_class.node_tree

            # remove default nodes
            material_class.node_tree.nodes.clear()

            # add new nodes  
            node_1 = node_tree.nodes.new('ShaderNodeOutputMaterial')
            node_2= node_tree.nodes.new('ShaderNodeBrightContrast')

            # link nodes
            node_tree.links.new(node_1.inputs[0], node_2.outputs[0])

            if class_name.split('_')[0] == 'background':
                node_2.inputs[0].default_value = (1, 1, 1, 1)
            else:
                node_2.inputs[0].default_value = ((i + 1)/255., 0., 0., 1)

            self.my_material[material_name] =  material_class


    def addNOCSMaterial(self):
        material_name = 'coord_color'
        mat = (bpy.data.materials.get(material_name) or bpy.data.materials.new(material_name))

        mat.use_nodes = True
        node_tree = mat.node_tree
        nodes = node_tree.nodes
        nodes.clear()        

        links = node_tree.links
        links.clear()

        vcol_R = nodes.new(type="ShaderNodeVertexColor")
        vcol_R.layer_name = "Col_R" # the vertex color layer name
        vcol_G = nodes.new(type="ShaderNodeVertexColor")
        vcol_G.layer_name = "Col_G" # the vertex color layer name
        vcol_B = nodes.new(type="ShaderNodeVertexColor")
        vcol_B.layer_name = "Col_B" # the vertex color layer name

        node_Output = node_tree.nodes.new('ShaderNodeOutputMaterial')
        node_Emission = node_tree.nodes.new('ShaderNodeEmission')
        node_LightPath = node_tree.nodes.new('ShaderNodeLightPath')
        node_Mix = node_tree.nodes.new('ShaderNodeMixShader')
        node_Combine = node_tree.nodes.new(type="ShaderNodeCombineRGB")


        # make links
        node_tree.links.new(vcol_R.outputs[1], node_Combine.inputs[0])
        node_tree.links.new(vcol_G.outputs[1], node_Combine.inputs[1])
        node_tree.links.new(vcol_B.outputs[1], node_Combine.inputs[2])
        node_tree.links.new(node_Combine.outputs[0], node_Emission.inputs[0])

        node_tree.links.new(node_LightPath.outputs[0], node_Mix.inputs[0])
        node_tree.links.new(node_Emission.outputs[0], node_Mix.inputs[2])
        node_tree.links.new(node_Mix.outputs[0], node_Output.inputs[0])

        self.my_material[material_name] = mat


    def addNormalMaterial(self):
        material_name = 'normal'
        mat = (bpy.data.materials.get(material_name) or bpy.data.materials.new(material_name))
        mat.use_nodes = True
        node_tree = mat.node_tree
        nodes = node_tree.nodes
        nodes.clear()
            
        links = node_tree.links
        links.clear()
            
        # Nodes:
        new_node = nodes.new(type='ShaderNodeMath')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (151.59744262695312, 854.5482177734375)
        new_node.name = 'Math'
        new_node.operation = 'MULTIPLY'
        new_node.select = False
        new_node.use_clamp = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = 0.5
        new_node.inputs[1].default_value = 1.0
        new_node.inputs[2].default_value = 0.0
        new_node.outputs[0].default_value = 0.0

        new_node = nodes.new(type='ShaderNodeLightPath')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (602.9912719726562, 1046.660888671875)
        new_node.name = 'Light Path'
        new_node.select = False
        new_node.width = 140.0
        new_node.outputs[0].default_value = 0.0
        new_node.outputs[1].default_value = 0.0
        new_node.outputs[2].default_value = 0.0
        new_node.outputs[3].default_value = 0.0
        new_node.outputs[4].default_value = 0.0
        new_node.outputs[5].default_value = 0.0
        new_node.outputs[6].default_value = 0.0
        new_node.outputs[7].default_value = 0.0
        new_node.outputs[8].default_value = 0.0
        new_node.outputs[9].default_value = 0.0
        new_node.outputs[10].default_value = 0.0
        new_node.outputs[11].default_value = 0.0
        new_node.outputs[12].default_value = 0.0

        new_node = nodes.new(type='ShaderNodeOutputMaterial')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.is_active_output = True
        new_node.location = (1168.93017578125, 701.84033203125)
        new_node.name = 'Material Output'
        new_node.select = False
        new_node.target = 'ALL'
        new_node.width = 140.0
        new_node.inputs[2].default_value = [0.0, 0.0, 0.0]

        new_node = nodes.new(type='ShaderNodeBsdfTransparent')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (731.72900390625, 721.4832763671875)
        new_node.name = 'Transparent BSDF'
        new_node.select = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = [1.0, 1.0, 1.0, 1.0]

        new_node = nodes.new(type='ShaderNodeCombineXYZ')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (594.4229736328125, 602.9271240234375)
        new_node.name = 'Combine XYZ'
        new_node.select = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = 0.0
        new_node.inputs[1].default_value = 0.0
        new_node.inputs[2].default_value = 0.0
        new_node.outputs[0].default_value = [0.0, 0.0, 0.0]

        new_node = nodes.new(type='ShaderNodeMixShader')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (992.7239990234375, 707.2142333984375)
        new_node.name = 'Mix Shader'
        new_node.select = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = 0.5

        new_node = nodes.new(type='ShaderNodeEmission')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (774.0802612304688, 608.2547607421875)
        new_node.name = 'Emission'
        new_node.select = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = [1.0, 1.0, 1.0, 1.0]
        new_node.inputs[1].default_value = 1.0

        new_node = nodes.new(type='ShaderNodeSeparateXYZ')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (-130.12167358398438, 558.1497802734375)
        new_node.name = 'Separate XYZ'
        new_node.select = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[0].default_value = 0.0
        new_node.outputs[1].default_value = 0.0
        new_node.outputs[2].default_value = 0.0

        new_node = nodes.new(type='ShaderNodeMath')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (162.43240356445312, 618.8094482421875)
        new_node.name = 'Math.002'
        new_node.operation = 'MULTIPLY'
        new_node.select = False
        new_node.use_clamp = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = 0.5
        new_node.inputs[1].default_value = 1.0
        new_node.inputs[2].default_value = 0.0
        new_node.outputs[0].default_value = 0.0

        new_node = nodes.new(type='ShaderNodeMath')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (126.8158187866211, 364.5539855957031)
        new_node.name = 'Math.001'
        new_node.operation = 'MULTIPLY'
        new_node.select = False
        new_node.use_clamp = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = 0.5
        new_node.inputs[1].default_value = -1.0
        new_node.inputs[2].default_value = 0.0
        new_node.outputs[0].default_value = 0.0

        new_node = nodes.new(type='ShaderNodeVectorTransform')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.convert_from = 'WORLD'
        new_node.convert_to = 'CAMERA'
        new_node.location = (-397.0209045410156, 594.7037353515625)
        new_node.name = 'Vector Transform'
        new_node.select = False
        new_node.vector_type = 'VECTOR'
        new_node.width = 140.0
        new_node.inputs[0].default_value = [0.5, 0.5, 0.5]
        new_node.outputs[0].default_value = [0.0, 0.0, 0.0]

        new_node = nodes.new(type='ShaderNodeNewGeometry')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (-651.8067016601562, 593.0455932617188)
        new_node.name = 'Geometry'
        new_node.width = 140.0
        new_node.outputs[0].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[1].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[2].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[3].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[4].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[5].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[6].default_value = 0.0
        new_node.outputs[7].default_value = 0.0
        new_node.outputs[8].default_value = 0.0

        # Links :

        links.new(nodes["Light Path"].outputs[0], nodes["Mix Shader"].inputs[0])    
        links.new(nodes["Separate XYZ"].outputs[0], nodes["Math"].inputs[0])    
        links.new(nodes["Separate XYZ"].outputs[1], nodes["Math.002"].inputs[0])    
        links.new(nodes["Separate XYZ"].outputs[2], nodes["Math.001"].inputs[0])    
        links.new(nodes["Vector Transform"].outputs[0], nodes["Separate XYZ"].inputs[0])    
        links.new(nodes["Combine XYZ"].outputs[0], nodes["Emission"].inputs[0])    
        links.new(nodes["Math"].outputs[0], nodes["Combine XYZ"].inputs[0])    
        links.new(nodes["Math.002"].outputs[0], nodes["Combine XYZ"].inputs[1])    
        links.new(nodes["Math.001"].outputs[0], nodes["Combine XYZ"].inputs[2])    
        links.new(nodes["Transparent BSDF"].outputs[0], nodes["Mix Shader"].inputs[1])    
        links.new(nodes["Emission"].outputs[0], nodes["Mix Shader"].inputs[2])    
        links.new(nodes["Mix Shader"].outputs[0], nodes["Material Output"].inputs[0])    
        links.new(nodes["Geometry"].outputs[1], nodes["Vector Transform"].inputs[0])    

        self.my_material[material_name] = mat


    def addMaterialLib(self):
        mat_specular_list = []
        mat_transparent_list = []
        mat_diffuse_list = []
        mat_background_list = []
        for mat in bpy.data.materials:
            name = mat.name
            name_class = name.split('_')[0]
            if name_class in material_class_instance_pairs['specular']:
                mat_specular_list.append(mat)
            if name_class in material_class_instance_pairs['transparent']:
                mat_transparent_list.append(mat)
            if name_class in material_class_instance_pairs['diffuse']:
                mat_diffuse_list.append(mat)
            if name_class in material_class_instance_pairs['background']:
                mat_background_list.append(mat)

        self.my_material['specular'] = mat_specular_list
        self.my_material['transparent'] = mat_transparent_list
        self.my_material['background'] = mat_background_list
        self.my_material['diffuse'] = mat_diffuse_list


    def setCamera(self, quaternion, translation, fov, baseline_distance):
        self.camera_l.data.angle = fov
        self.camera_r.data.angle = self.camera_l.data.angle
        cx = translation[0]
        cy = translation[1]
        cz = translation[2]

        self.camera_l.location[0] = cx
        self.camera_l.location[1] = cy 
        self.camera_l.location[2] = cz

        self.camera_l.rotation_mode = 'QUATERNION'
        self.camera_l.rotation_quaternion[0] = quaternion[0]
        self.camera_l.rotation_quaternion[1] = quaternion[1]
        self.camera_l.rotation_quaternion[2] = quaternion[2]
        self.camera_l.rotation_quaternion[3] = quaternion[3]

        self.camera_r.rotation_mode = 'QUATERNION'
        self.camera_r.rotation_quaternion[0] = quaternion[0]
        self.camera_r.rotation_quaternion[1] = quaternion[1]
        self.camera_r.rotation_quaternion[2] = quaternion[2]
        self.camera_r.rotation_quaternion[3] = quaternion[3]
        cx, cy, cz = cameraLPosToCameraRPos(quaternion, (cx, cy, cz), baseline_distance)
        self.camera_r.location[0] = cx
        self.camera_r.location[1] = cy 
        self.camera_r.location[2] = cz


    def setLighting(self):
        # emitter        
        #self.light_emitter.location = self.camera_r.location
        self.light_emitter.location = self.camera_l.location + 0.51 * (self.camera_r.location - self.camera_l.location)
        self.light_emitter.rotation_mode = 'QUATERNION'
        self.light_emitter.rotation_quaternion = self.camera_r.rotation_quaternion

        # emitter setting
        bpy.context.view_layer.objects.active = None
        # bpy.ops.object.select_all(action="DESELECT")
        self.render_context.engine = 'CYCLES'
        self.light_emitter.select_set(True)
        self.light_emitter.data.use_nodes = True
        self.light_emitter.data.type = "POINT"
        self.light_emitter.data.shadow_soft_size = 0.001
        random_energy = random.uniform(LIGHT_EMITTER_ENERGY * 0.9, LIGHT_EMITTER_ENERGY * 1.1)
        self.light_emitter.data.energy = random_energy

        # remove default node
        light_emitter = bpy.data.objects["light_emitter"].data
        light_emitter.node_tree.nodes.clear()

        # add new nodes
        light_output = light_emitter.node_tree.nodes.new("ShaderNodeOutputLight")
        node_1 = light_emitter.node_tree.nodes.new("ShaderNodeEmission")
        node_2 = light_emitter.node_tree.nodes.new("ShaderNodeTexImage")
        node_3 = light_emitter.node_tree.nodes.new("ShaderNodeMapping")
        node_4 = light_emitter.node_tree.nodes.new("ShaderNodeVectorMath")
        node_5 = light_emitter.node_tree.nodes.new("ShaderNodeSeparateXYZ")
        node_6 = light_emitter.node_tree.nodes.new("ShaderNodeTexCoord")

        # link nodes
        light_emitter.node_tree.links.new(light_output.inputs[0], node_1.outputs[0])
        light_emitter.node_tree.links.new(node_1.inputs[0], node_2.outputs[0])
        light_emitter.node_tree.links.new(node_2.inputs[0], node_3.outputs[0])
        light_emitter.node_tree.links.new(node_3.inputs[0], node_4.outputs[0])
        light_emitter.node_tree.links.new(node_4.inputs[0], node_6.outputs[1])
        light_emitter.node_tree.links.new(node_4.inputs[1], node_5.outputs[2])
        light_emitter.node_tree.links.new(node_5.inputs[0], node_6.outputs[1])

        # set parameter of nodes
        node_1.inputs[1].default_value = 1.0        # scale
        node_2.extension = 'CLIP'
        # node_2.interpolation = 'Cubic'

        node_3.inputs[1].default_value[0] = 0.5
        node_3.inputs[1].default_value[1] = 0.5
        node_3.inputs[1].default_value[2] = 0
        node_3.inputs[2].default_value[0] = 0
        node_3.inputs[2].default_value[1] = 0
        node_3.inputs[2].default_value[2] = 0.05

        # scale of pattern
        node_3.inputs[3].default_value[0] = 0.6
        node_3.inputs[3].default_value[1] = 0.85
        node_3.inputs[3].default_value[2] = 0
        node_4.operation = 'DIVIDE'

        # pattern path
        node_2.image = self.pattern


    def lightModeSelect(self, light_mode):
        if light_mode == "RGB":
            self.light_emitter.hide_render = True
            # set the environment map energy
            random_energy = random.uniform(LIGHT_ENV_MAP_ENERGY_RGB * 0.8, LIGHT_ENV_MAP_ENERGY_RGB * 1.2)
            bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = random_energy

        elif light_mode == "IR":
            self.light_emitter.hide_render = False
            # set the environment map energy
            random_energy = random.uniform(LIGHT_ENV_MAP_ENERGY_IR * 0.8, LIGHT_ENV_MAP_ENERGY_IR * 1.2)
            bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = random_energy
        
        elif light_mode == "Mask" or light_mode == "NOCS" or light_mode == "Normal":
            self.light_emitter.hide_render = True
            bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0

        else:
            print("Not support the mode!")    


    def outputModeSelect(self, output_mode):
        if output_mode == "RGB":
            self.render_context.image_settings.file_format = 'PNG'
            self.render_context.image_settings.compression = 0
            self.render_context.image_settings.color_mode = 'RGB'
            self.render_context.image_settings.color_depth = '8'
            bpy.context.scene.view_settings.view_transform = 'Filmic'
            bpy.context.scene.render.filter_size = 1.5
            self.render_context.resolution_x = 1280
            self.render_context.resolution_y = 720
        elif output_mode == "IR":
            self.render_context.image_settings.file_format = 'PNG'
            self.render_context.image_settings.compression = 0
            self.render_context.image_settings.color_mode = 'BW'
            self.render_context.image_settings.color_depth = '8'
            bpy.context.scene.view_settings.view_transform = 'Filmic'
            bpy.context.scene.render.filter_size = 1.5
            self.render_context.resolution_x = 1280
            self.render_context.resolution_y = 720
        elif output_mode == "Mask":
            self.render_context.image_settings.file_format = 'OPEN_EXR'
            self.render_context.image_settings.color_mode = 'RGB'
            bpy.context.scene.view_settings.view_transform = 'Raw'
            bpy.context.scene.render.filter_size = 0
            self.render_context.resolution_x = 640
            self.render_context.resolution_y = 360
        elif output_mode == "NOCS":
            # self.render_context.image_settings.file_format = 'OPEN_EXR'
            self.render_context.image_settings.file_format = 'PNG'            
            self.render_context.image_settings.color_mode = 'RGB'
            self.render_context.image_settings.color_depth = '8'
            bpy.context.scene.view_settings.view_transform = 'Raw'
            bpy.context.scene.render.filter_size = 0
            self.render_context.resolution_x = 640
            self.render_context.resolution_y = 360
        elif output_mode == "Normal":
            self.render_context.image_settings.file_format = 'OPEN_EXR'
            self.render_context.image_settings.color_mode = 'RGB'
            bpy.context.scene.view_settings.view_transform = 'Raw'
            bpy.context.scene.render.filter_size = 1.5
            self.render_context.resolution_x = 640
            self.render_context.resolution_y = 360
        else:
            print("Not support the mode!")    


    def renderEngineSelect(self, engine_mode):

        if engine_mode == "CYCLES":
            self.render_context.engine = 'CYCLES'
            bpy.context.scene.cycles.progressive = 'BRANCHED_PATH'
            bpy.context.scene.cycles.use_denoising = True
            bpy.context.scene.cycles.denoiser = 'NLM'
            bpy.context.scene.cycles.film_exposure = 1.0
            bpy.context.scene.cycles.aa_samples = 64 #32

            ## Set the device_type
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # or "OPENCL"
            ## Set the device and feature set
            # bpy.context.scene.cycles.device = "CPU"

            ## get_devices() to let Blender detects GPU device
            cuda_devices, _ = bpy.context.preferences.addons["cycles"].preferences.get_devices()
            print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
            for d in bpy.context.preferences.addons["cycles"].preferences.devices:
                d["use"] = 1 # Using all devices, include GPU and CPU
                print(d["name"], d["use"])
            '''
            '''
            device_list = DEVICE_LIST
            activated_gpus = []
            for i, device in enumerate(cuda_devices):
                if (i in device_list):
                    device.use = True
                    activated_gpus.append(device.name)
                else:
                    device.use = False


        elif engine_mode == "EEVEE":
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        else:
            print("Not support the mode!")    


    def addBackground(self, size, position, scale):
        # set the material of background    
        material_name = "default_background"

        # test if material exists
        # if it does not exist, create it:
        material_background = (bpy.data.materials.get(material_name) or 
            bpy.data.materials.new(material_name))

        # enable 'Use nodes'
        material_background.use_nodes = True
        node_tree = material_background.node_tree

        # remove default nodes
        material_background.node_tree.nodes.clear()
        # material_background.node_tree.nodes.remove(material_background.node_tree.nodes.get('Principled BSDF')) #title of the existing node when materials.new
        # material_background.node_tree.nodes.remove(material_background.node_tree.nodes.get('Material Output')) #title of the existing node when materials.new

        # add new nodes  
        node_1 = node_tree.nodes.new('ShaderNodeOutputMaterial')
        node_2 = node_tree.nodes.new('ShaderNodeBsdfPrincipled')
        node_3 = node_tree.nodes.new('ShaderNodeTexImage')

        # link nodes
        node_tree.links.new(node_1.inputs[0], node_2.outputs[0])
        node_tree.links.new(node_2.inputs[0], node_3.outputs[0])

        # add texture image
        node_3.image = bpy.data.images.load(filepath=os.path.join(working_root, "texture/texture_0.jpg"))
        self.my_material['default_background'] = material_background

        # add background plane
        for i in range(-2, 3, 1):
            for j in range(-2, 3, 1):
                position_i_j = (i * size + position[0], j * size + position[1], position[2])
                bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, align='WORLD', location=position_i_j, scale=scale)
                bpy.ops.rigidbody.object_add()
                bpy.context.object.rigid_body.type = 'PASSIVE'
                bpy.context.object.rigid_body.collision_shape = 'BOX'
        for i in range(-2, 3, 1):
            for j in [-2, 2]:
                position_i_j = (i * size + position[0], j * size + position[1], position[2] - 0.25)
                rotation_elur = (math.pi / 2., 0., 0.)
                bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, align='WORLD', location=position_i_j, rotation = rotation_elur)
                bpy.ops.rigidbody.object_add()
                bpy.context.object.rigid_body.type = 'PASSIVE'
                bpy.context.object.rigid_body.collision_shape = 'BOX'    
        for j in range(-2, 3, 1):
            for i in [-2, 2]:
                position_i_j = (i * size + position[0], j * size + position[1], position[2] - 0.25)
                rotation_elur = (0, math.pi / 2, 0)
                bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, align='WORLD', location=position_i_j, rotation = rotation_elur)
                bpy.ops.rigidbody.object_add()
                bpy.context.object.rigid_body.type = 'PASSIVE'
                bpy.context.object.rigid_body.collision_shape = 'BOX'        
        count = 0
        for obj in bpy.data.objects:
            if obj.type == "MESH":
                obj.name = "background_" + str(count)
                obj.data.name = "background_" + str(count)
                obj.active_material = material_background
                count += 1

        self.background_added = True


    def clearModel(self):
        '''
        # delete all meshes
        for item in bpy.data.meshes:
            bpy.data.meshes.remove(item)
        for item in bpy.data.materials:
            bpy.data.materials.remove(item)
        '''

        # remove all objects except background
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and not obj.name.split('_')[0] == 'background':
                bpy.data.meshes.remove(obj.data)
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and not obj.name.split('_')[0] == 'background':
                bpy.data.objects.remove(obj, do_unlink=True)

        # remove all default material
        for mat in bpy.data.materials:
            name = mat.name.split('.')
            if name[0] == 'Material':
                bpy.data.materials.remove(mat)


    def loadModel(self, file_path):
        self.model_loaded = True
        try:
            if file_path.endswith('obj'):
                bpy.ops.import_scene.obj(filepath=file_path)
            elif file_path.endswith('3ds'):
                bpy.ops.import_scene.autodesk_3ds(filepath=file_path)
            elif file_path.endswith('dae'):
                # Must install OpenCollada. Please read README.md
                bpy.ops.wm.collada_import(filepath=file_path)
            else:
                self.model_loaded = False
                raise Exception("Loading failed: %s" % (file_path))
        except Exception:
            self.model_loaded = False


    def render(self, image_name="tmp", image_path=RENDERING_PATH):
        # Render the object
        if not self.model_loaded:
            print("Model not loaded.")
            return      

        if self.render_mode == "IR":
            bpy.context.scene.use_nodes = False
            # set light and render mode
            self.lightModeSelect("IR")
            self.outputModeSelect("IR")
            self.renderEngineSelect("CYCLES")

        elif self.render_mode == 'RGB':
            bpy.context.scene.use_nodes = False
            # set light and render mode
            self.lightModeSelect("RGB")
            self.outputModeSelect("RGB")
            self.renderEngineSelect("CYCLES")

        elif self.render_mode == "Mask":
            bpy.context.scene.use_nodes = False
            # set light and render mode
            self.lightModeSelect("Mask")
            self.outputModeSelect("Mask")
            # self.renderEngineSelect("EEVEE")
            self.renderEngineSelect("CYCLES")
            bpy.context.scene.cycles.use_denoising = False
            bpy.context.scene.cycles.aa_samples = 1

        elif self.render_mode == "NOCS":
            bpy.context.scene.use_nodes = False
            # set light and render mode
            self.lightModeSelect("NOCS")
            self.outputModeSelect("NOCS")
            # self.renderEngineSelect("EEVEE")
            self.renderEngineSelect("CYCLES")
            bpy.context.scene.cycles.use_denoising = False
            bpy.context.scene.cycles.aa_samples = 1

        elif self.render_mode == "Normal":
            bpy.context.scene.use_nodes = True
            self.fileOutput.base_path = image_path
            self.fileOutput.file_slots[0].path = image_name[:5] + 'depth_#'

            # set light and render mode
            self.lightModeSelect("Normal")
            self.outputModeSelect("Normal")
            # self.renderEngineSelect("EEVEE")
            self.renderEngineSelect("CYCLES")
            bpy.context.scene.cycles.use_denoising = False
            bpy.context.scene.cycles.aa_samples = 32

        else:
            print("The render mode is not supported")
            return 

        bpy.context.scene.render.filepath = os.path.join(image_path, image_name)
        bpy.ops.render.render(write_still=True)  # save straight to file


    def set_material_randomize_mode(self, class_material_pairs, mat_randomize_mode, instance, material_type_in_mixed_mode):
        if mat_randomize_mode == 'transparent' and instance.name.split('_')[1] in class_material_pairs['transparent']:
            print(instance.name, 'material mode: transparent')            
            instance.data.materials.clear()
            instance.active_material = random.sample(self.my_material['transparent'], 1)[0]

        elif mat_randomize_mode == 'specular' and instance.name.split('_')[1] in class_material_pairs['specular']:
            print(instance.name, 'material mode: specular')
            material = random.sample(self.my_material['specular'], 1)[0]
            set_modify_material(instance, material)     

        elif mat_randomize_mode == 'mixed':
            if material_type_in_mixed_mode == 'diffuse' and instance.name.split('_')[1] in class_material_pairs['diffuse']:
                print(instance.name, 'material mode: diffuse')
                material = random.sample(self.my_material['diffuse'], 1)[0]
                set_modify_material(instance, material)
            elif material_type_in_mixed_mode == 'transparent' and instance.name.split('_')[1] in class_material_pairs['transparent']:
                print(instance.name, 'material mode: transparent')
                instance.data.materials.clear()
                instance.active_material = random.sample(self.my_material['transparent'], 1)[0]
            elif material_type_in_mixed_mode == 'specular' and instance.name.split('_')[1] in class_material_pairs['specular']:
                print(instance.name, 'material mode: specular')
                material = random.sample(self.my_material['specular'], 1)[0]
                set_modify_material(instance, material)       
            else:
                print(instance.name, 'material mode: raw')
                set_modify_raw_material(instance)
        else:
            print(instance.name, 'material mode: raw')
            set_modify_raw_material(instance)


    def get_instance_pose(self):
        instance_pose = {}
        bpy.context.view_layer.update()
        cam = self.camera_l
        mat_rot_x = Matrix.Rotation(math.radians(180.0), 4, 'X')
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and not obj.name.split('_')[0] == 'background':
                instance_id = obj.name.split('_')[0]
                mat_rel = cam.matrix_world.inverted() @ obj.matrix_world
                # location
                relative_location = [mat_rel.translation[0],
                                     - mat_rel.translation[1],
                                     - mat_rel.translation[2]]
                # rotation
                # relative_rotation_euler = mat_rel.to_euler() # must be converted from radians to degrees
                relative_rotation_quat = [mat_rel.to_quaternion()[0],
                                          mat_rel.to_quaternion()[1],
                                          mat_rel.to_quaternion()[2],
                                          mat_rel.to_quaternion()[3]]
                quat_x = [0, 1, 0, 0]
                quat = quanternion_mul(quat_x, relative_rotation_quat)
                quat = [quat[0], - quat[1], - quat[2], - quat[3]]
                instance_pose[str(instance_id)] = [quat, relative_location]

        return instance_pose


    def check_visible(self, threshold=(0.1, 0.9, 0.1, 0.9)):
        w_min, x_max, h_min, h_max = threshold
        visible_objects_list = []
        bpy.context.view_layer.update()
        cs, ce = self.camera_l.data.clip_start, self.camera_l.data.clip_end
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and not obj.name.split('_')[0] == 'background':
                obj_center = obj.matrix_world.translation
                co_ndc = world_to_camera_view(scene, self.camera_l, obj_center)
                if (w_min < co_ndc.x < x_max and
                    h_min < co_ndc.y < h_max and
                    cs < co_ndc.z <  ce):
                    obj.select_set(True)
                    visible_objects_list.append(obj)
                else:
                    obj.select_set(False)
        return visible_objects_list


###########################
# Main
###########################
selected_class = ['aeroplane', 'bottle', 'bowl', 'camera', 'can', 'car', 'mug']
g_shape_synset_name_pairs = copy.deepcopy(g_shape_synset_name_pairs_all)
g_shape_synset_name_pairs['00000000'] = 'other'
for item in g_shape_synset_name_pairs_all:
    if not g_shape_synset_name_pairs_all[item] in selected_class:
        g_shape_synset_name_pairs[item] = 'other'

g_synset_name_scale_pairs = {'aeroplane': [0.25, 0.31],
                             'bottle': [0.21, 0.27], 
                             'bowl': [0.15, 0.20],
                             'camera': [0.17, 0.23], 
                             'can': [0.13, 0.17],
                             'car': [0.21, 0.25], 
                             'mug': [0.13, 0.19],
                             'other': [0.13, 0.22]} 

g_synset_name_label_pairs = {'aeroplane': 7,
                             'bottle': 1,
                             'bowl': 2,   
                             'camera': 3,
                             'can': 4,
                             'car': 5,
                             'mug': 6,    
                             'other': 0}   

material_class_instance_pairs = {'specular': ['metal', 'porcelain'],
                                 'transparent': ['glass'],
                                 'diffuse': ['plastic', 'rubber'],
                                 'background': ['background']}

class_material_pairs = {'specular': ['bottle', 'bowl', 'can', 'mug', 'aeroplane', 'car', 'other'],
                        'transparent': ['bottle', 'bowl', 'mug'],
                        'diffuse': ['bottle', 'bowl', 'can', 'mug', 'camera', 'aeroplane', 'car', 'other']}

material_name_label_pairs = {'raw': 0,
                             'diffuse': 1,
                             'transparent': 2,
                             'specular': 3}


max_instance_num = 20

if not os.path.exists(output_root_path):
    os.makedirs(output_root_path)

# generate CAD model list
CAD_model_list = generate_CAD_model_list(CAD_model_root_path)

renderer = BlenderRenderer(viewport_size_x=camera_width, viewport_size_y=camera_height)
renderer.loadImages(env_map_path)
renderer.addEnvMap()
renderer.addBackground(background_size, background_position, background_scale)
renderer.addMaterialLib()
renderer.addMaskMaterial(max_instance_num)
renderer.addNOCSMaterial()
renderer.addNormalMaterial()

print(len(renderer.my_material['specular']))
print(len(renderer.my_material['transparent']))
print(len(renderer.my_material['diffuse']))
print(len(renderer.my_material['background']))


renderer.clearModel()
# set scene output path
path_scene = os.path.join(output_root_path, str(SCENE_NUM).zfill(5))
if not os.path.exists(path_scene):
    os.mkdir(path_scene)


# camera pose list, environment light list and background material_listz
quaternion_list = []
translation_list = []

# environment map list
env_map_id_list = []
rotation_elur_z_list = []

# background material list
background_material_list = []

for i in range(num_frame_per_scene):
    # generate camara pose list
    quaternion, translation = cameraPositionRandomize(start_point_range, look_at_range, up_range)
    quaternion_list.append(quaternion)
    translation_list.append(translation)

    # generate environment map list
    env_map_id_list.append(random.randint(0, len(renderer.env_map) - 1))
    rotation_elur_z_list.append(random.uniform(-math.pi, math.pi))
    # generate background material list 

    if my_material_randomize_mode == 'raw':
        background_material_list.append(renderer.my_material['default_background'])
        # bpy.data.objects['background'].active_material = renderer.my_material['default_background']
    else:
        material_selected = random.sample(renderer.my_material['background'], 1)[0]
        background_material_list.append(material_selected)
        # bpy.data.objects['background'].active_material = material_selected


# read objects from floder
instance_id = 1
meta_output = {}
#select_model_list = []
select_model_list_other = []
select_model_list_transparent = []
select_model_list_dis = []
select_number = 1

#for item in CAD_model_list:
#    if item in ['bottle', 'bowl', 'mug']:
#        select_number = min(15, len(CAD_model_list[item]))
#    else:
#        select_number = 3
#    test = random.sample(CAD_model_list[item], select_number)
#    for model in test:
#        select_model_list.append(model)
#select_model_list = random.sample(select_model_list, random.randint(4, 10))
#print("###################################")
#print(CAD_model_list['other'])
#print("###################################")

for item in CAD_model_list:
    if item in ['bottle', 'bowl', 'mug']:
        test = random.sample(CAD_model_list[item], select_number)
        for model in test:
            select_model_list_transparent.append(model)
    elif item in ['other']:
        test = random.sample(CAD_model_list[item], min(3, len(CAD_model_list[item])))
        for model in test:
            select_model_list_dis.append(model)
    else:
        test = random.sample(CAD_model_list[item], select_number)
        for model in test:
            select_model_list_other.append(model)

#select_model_list_transparent = random.sample(select_model_list_transparent, random.randint(2, 3))
select_model_list_other = random.sample(select_model_list_other, random.randint(1, 4))
select_model_list_dis = random.sample(select_model_list_dis, random.randint(1, 3))

select_model_list = select_model_list_transparent + select_model_list_other + select_model_list_dis

# for item in CAD_model_list:
#     select_model_list.append(CAD_model_list[item][0])


# select_model_list = random.sample(select_model_list, 1)
for model in select_model_list:
    instance_path = model[0]
    class_name = model[1]
    class_folder = model[0].split('/')[-3]
    instance_folder = model[0].split('/')[-2]
    instance_name = str(instance_id) + "_" + class_name + "_" + class_folder + "_" + instance_folder
    material_type_in_mixed_mode = generate_material_type(instance_name)

    # download CAD model and rename
    renderer.loadModel(instance_path)
    obj = bpy.data.objects['model']
    obj.name = instance_name
    obj.data.name = instance_name

    setModelPosition(obj, (-0.3, 0.3, -0.3, 0.3, background_position[2] + 0.2), instance_id)

    # set object as rigid body
    setRigidBody(obj)

    # set material
    renderer.set_material_randomize_mode(class_material_pairs, my_material_randomize_mode, obj, material_type_in_mixed_mode)

    # generate meta file
    class_scale = random.uniform(g_synset_name_scale_pairs[class_name][0], g_synset_name_scale_pairs[class_name][1])
    obj.scale = (class_scale, class_scale, class_scale)

    meta_output[str(instance_id)] = [str(g_synset_name_label_pairs[class_name]),
                                     class_folder, 
                                     instance_folder, 
                                     str(class_scale),
                                     str(material_name_label_pairs[material_type_in_mixed_mode])]
    instance_id += 1

# set the key frame
scene = bpy.data.scenes['Scene']
scene.frame_start = 0
scene.frame_end = 121

render_output_file = path_scene

for i in range(scene.frame_start, scene.frame_end + 1):
    scene.frame_set(i)
    if i == 120:
        break  


# Get the list of visible objects and the list of object pose corresponding to the camera list
visible_objects_list = []
instance_pose_list = []
visible_threshold = (0.03, 0.97, 0.05, 0.95) #(0.1, 0.9, 0.1, 0.9)
for i in range(num_frame_per_scene):
    # generate visible objects list
    renderer.setCamera(quaternion_list[i], translation_list[i], camera_fov, baseline_distance)
    visible_objects_list.append(renderer.check_visible(visible_threshold))
    # generate object pose list
    instance_pose_list.append(renderer.get_instance_pose())

# generate meta.txt
for i in range(num_frame_per_scene):
    # output the meta file
    path_meta = os.path.join(path_scene, str(i).zfill(4) + "_meta.txt")
    if os.path.exists(path_meta):
        os.remove(path_meta)
    
    file_write_obj = open(path_meta, 'w')
    for index in meta_output:
        file_write_obj.write(index)
        file_write_obj.write(' ')
        for item in meta_output[index]:
            file_write_obj.write(item)
            file_write_obj.write(' ')
        for item in instance_pose_list[i][index]:
            for var in item:
                file_write_obj.write(str(var))
                file_write_obj.write(' ')
        file_write_obj.write('\n')
    file_write_obj.close()


# render IR image and RGB image
if render_mode_list['IR'] or render_mode_list['RGB']:
    for i in range(num_frame_per_scene):
        set_visiable_objects(visible_objects_list[i])
        # renderer.set_material_randomize_mode(my_material_randomize_mode)
        renderer.setCamera(quaternion_list[i], translation_list[i], camera_fov, baseline_distance)
        renderer.setLighting()
        renderer.setEnvMap(env_map_id_list[i], rotation_elur_z_list[i])
        for obj in bpy.data.objects:
            if obj.type == "MESH" and obj.name.split('_')[0] == 'background':
                obj.active_material = background_material_list[i]

        # render IR image            
        if render_mode_list['IR']:
            renderer.render_mode = "IR"
            camera = bpy.data.objects['camera_l']
            scene.camera = camera
            save_path = render_output_file
            save_name = str(i).zfill(4) + '_ir_l'
            renderer.render(save_name, save_path)

            camera = bpy.data.objects['camera_r']
            scene.camera = camera
            save_path = render_output_file
            save_name = str(i).zfill(4) + '_ir_r'
            renderer.render(save_name, save_path)
        
        # render RGB image
        if render_mode_list['RGB']:
            renderer.render_mode = "RGB"
            camera = bpy.data.objects['camera_l']
            scene.camera = camera
            save_path = render_output_file
            save_name = str(i).zfill(4) + '_color'
            renderer.render(save_name, save_path)
    
# render mask map and depth map
if render_mode_list['Mask']:
    # set instance mask as material
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.data.materials.clear()
            material_name = "mask_" + obj.name.split('_')[0]
            obj.active_material = renderer.my_material[material_name]
    
    # render mask map and depth map
    for i in range(num_frame_per_scene):
        set_visiable_objects(visible_objects_list[i])
        renderer.setCamera(quaternion_list[i], translation_list[i], camera_fov, baseline_distance)
        # renderer.light_env_on_id = light_selected_id_list[i]
        # renderer.setLighting(size=num_env_light, position_z=env_light_z_list[i])
        renderer.render_mode = "Mask"
        camera = bpy.data.objects['camera_l']
        scene.camera = camera
        save_path = render_output_file
        save_name = str(i).zfill(4) + '_mask'
        renderer.render(save_name, save_path)

# render normal map
if render_mode_list['Normal']:
    # set normal as material
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
            obj.active_material = renderer.my_material["normal"]

    # render normal map
    for i in range(num_frame_per_scene):
        set_visiable_objects(visible_objects_list[i])
        renderer.setCamera(quaternion_list[i], translation_list[i], camera_fov, baseline_distance)
        # renderer.light_env_on_id = light_selected_id_list[i]
        # renderer.setLighting(size=num_env_light, position_z=env_light_z_list[i])
        renderer.render_mode = "Normal"
        camera = bpy.data.objects['camera_l']
        scene.camera = camera
        save_path = render_output_file
        save_name = str(i).zfill(4) + '_normal'
        renderer.render(save_name, save_path)

# render mocs map
if render_mode_list['NOCS']:
    # # set nocs vertex color 
    # start_time = time.time()
    # print("###################################")
    # # print(len(bpy.data.objects))
    # # print(len(bpy.data.meshes))

    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name.split('_')[0] == 'background':
            start_time_obj = time.time()
            vertex_colors = obj.data.vertex_colors
            # remove exists vertex colors
            while vertex_colors:
                vertex_colors.remove(vertex_colors[0])
            obj.data.update()
            # create new vertex color layer
            obj.data.vertex_colors.new(name='Col_R', do_init=False)
            obj.data.vertex_colors.new(name='Col_G', do_init=False)
            obj.data.vertex_colors.new(name='Col_B', do_init=False)
            vcol_layer_r = obj.data.vertex_colors['Col_R']
            vcol_layer_g = obj.data.vertex_colors['Col_G']
            vcol_layer_b = obj.data.vertex_colors['Col_B']

            count = 0
            start_time_loop = time.time()
            for loop_index, loop in enumerate(obj.data.loops):
                vcol_layer_r.data[loop_index].color = Vector([0, 0, 0, 1])
                vcol_layer_g.data[loop_index].color = Vector([0, 0, 0, 1])
                vcol_layer_b.data[loop_index].color = Vector([0, 0, 0, 1])
                count += 1
            end_time_obj = time.time()
            obj.data.vertex_colors.active = vcol_layer_r
            obj.data.update()
            print(obj.name, ' time: ', end_time_obj - start_time_obj, 'mean time: ', (end_time_obj - start_time_loop)/count)


        if obj.type == 'MESH' and not obj.name.split('_')[0] == 'background':
            start_time_obj = time.time()
            vertex_colors = obj.data.vertex_colors
            # remove exists vertex colors
            while vertex_colors:
                vertex_colors.remove(vertex_colors[0])
            obj.data.update()

            # create new vertex color layer
            obj.data.vertex_colors.new(name='Col_R', do_init=True)
            obj.data.vertex_colors.new(name='Col_G', do_init=True)
            obj.data.vertex_colors.new(name='Col_B', do_init=True)
            vcol_layer_r = obj.data.vertex_colors['Col_R']
            vcol_layer_g = obj.data.vertex_colors['Col_G']
            vcol_layer_b = obj.data.vertex_colors['Col_B']


            count = 0
            start_time_loop = time.time()
            for loop_index, loop in enumerate(obj.data.loops):
                loop_vert_index = loop.vertex_index
                # here the scale is manually set for the cube to normalize it within [-0.5, 0.5]
                scale = 1
                color_x = scale * obj.data.vertices[loop_vert_index].co.x + 0.5
                color_y = scale * obj.data.vertices[loop_vert_index].co.y + 0.5
                color_z = scale * obj.data.vertices[loop_vert_index].co.z + 0.5
                vcol_layer_r.data[loop_index].color = Vector([0, 0, 0, color_x])
                vcol_layer_g.data[loop_index].color = Vector([0, 0, 0, color_y])
                vcol_layer_b.data[loop_index].color = Vector([0, 0, 0, 1 - color_z])
                count += 1
            end_time_obj = time.time()
            obj.data.vertex_colors.active = vcol_layer_r
            obj.data.update()
            print(obj.name, ' time: ', end_time_obj - start_time_obj, 'mean time: ', (end_time_obj - start_time_loop)/count)
            


    # end_time = time.time()
    # print('time: ', end_time - start_time)
    # print("###################################")

    # set nocs map as material
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
            obj.active_material = renderer.my_material["coord_color"]

    # render nocs map
    for i in range(num_frame_per_scene):
        set_visiable_objects(visible_objects_list[i])
        renderer.setCamera(quaternion_list[i], translation_list[i], camera_fov, baseline_distance)
        # renderer.light_env_on_id = light_selected_id_list[i]
        # renderer.setLighting(size=num_env_light, position_z=env_light_z_list[i])
        renderer.render_mode = "NOCS"
        camera = bpy.data.objects['camera_l']
        scene.camera = camera
        save_path = render_output_file
        save_name = str(i).zfill(4) + '_coord'
        renderer.render(save_name, save_path)
context = bpy.context
for ob in context.selected_objects:
    ob.animation_data_clear()

print(bpy.data.materials) 
print(len(bpy.data.materials))
