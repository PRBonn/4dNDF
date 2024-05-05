#############################
#  To use this script, you may need to install some extra libraries.
#  pip install opencv-python openexr
#############################

import open3d as o3d
import cv2
import numpy as np
from tqdm import tqdm

import sys
import os

import OpenEXR
import Imath
import quaternion
from PIL import Image

def readFormFolder(folder_path, start_str, end_str):
    files = os.listdir(folder_path)
    files = [f for f in files if f.startswith(start_str) and f.endswith(end_str)]
    
    start_char = len(start_str)
    files = sorted(files, key=lambda f: int(f[start_char:start_char+4]))
    
    return files

def readPosesFile(filename):
    pose_file = open(filename)
    poses = []
    for line in pose_file:
        values = np.matrix([float(v) for v in line.strip().split()])
        t = values[0,1:4].transpose()
        q = np.quaternion(values[0,7], values[0,4], values[0,5], values[0,6])
        R = quaternion.as_rotation_matrix(q)
        T_0 = np.block([[R,t],[0, 0, 0, 1]])
        poses.append(T_0) 

    pose_file.close()
    return poses

def readEXR(filename):
    # Open the EXR file
    file = OpenEXR.InputFile(filename)
    # Get the size of the depth map
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channel = file.channel('Y', FLOAT)
    data = np.frombuffer(channel, dtype=np.float32)
    data.shape = (height, width)

    return data

def readMask(filename):
    img = Image.open(filename)
    img_array = np.array(img)
    mask = (img_array == [0, 0, 0]).all(axis=2)
    return mask

def convertToPointCloud(depth, fx, fy, cx, cy):
    # Convert the depth image to a point cloud
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    z = depth.copy()
    z[z==np.inf] = 11.0
    z[z==-np.inf] = 11.0
    z[z>10] = 0.0
    x = (c - cx) * z / fx
    y = (r - cy) * z / fy
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).transpose()
    return points

dataset_path = '/home/starry/Data/cofusion/car4'
fx = 564.3
fy = 564.3
cx = 480
cy = 270

if __name__ == "__main__":

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        sys.exit("Please enter the dataset path")
    
    depth_original_path = dataset_path + '/depth_original/'
    depth_noise_path = dataset_path + '/depth_noise/'
    mask_color_path = dataset_path + '/mask_color/'
    pose_file_path = dataset_path + '/trajectories/gt-cam-0.txt'
    
    static_output_path = dataset_path + '/ply_static'
    noise_outout_path = dataset_path + '/ply_noise' 
    
    if not os.path.exists(static_output_path):
        os.makedirs(static_output_path)
    if not os.path.exists(noise_outout_path):
        os.makedirs(noise_outout_path)
    
    depth_original_files = readFormFolder(depth_original_path, 'Depth', '.exr')
    depth_noise_files = readFormFolder(depth_noise_path, 'Depth', '.exr')
    mask_files = readFormFolder(mask_color_path, '', '.png')
    poses = readPosesFile(pose_file_path)

    for i in tqdm(range(len(poses)), desc='loading'):
        depth_original_data_source = readEXR(depth_original_path+depth_original_files[i])
        depth_noise_data = readEXR(depth_noise_path+depth_noise_files[i])
        depth_original_data = depth_original_data_source.copy()
        
        # remove dynamics, used as the groundtruth
        mask = readMask(mask_color_path+mask_files[i])
        depth_original_data[~mask] = 11.0
        origian_points = convertToPointCloud(depth_original_data, fx, fy, cx, cy)
        static_mask = (origian_points[:,0] == 0) & (origian_points[:,1] == 0) & (origian_points[:,2] == 0)
        static_points = origian_points[~static_mask]
        
        static_points_output = o3d.geometry.PointCloud()
        static_points_output.points = o3d.utility.Vector3dVector(static_points)
        o3d.io.write_point_cloud(static_output_path +'/' + str(i+1) + '.ply', static_points_output)
        
        # noisy points, used as the input
        noise_points = convertToPointCloud(depth_noise_data, fx, fy, cx, cy)
        noise_mask = (noise_points[:,0] == 0) & (noise_points[:,1] == 0) & (noise_points[:,2] == 0)
        noise_points = noise_points[~noise_mask]
        
        noise_points_output = o3d.geometry.PointCloud()
        noise_points_output.points = o3d.utility.Vector3dVector(noise_points)
        o3d.io.write_point_cloud(noise_outout_path +'/' + str(i+1) + '.ply', noise_points_output)
