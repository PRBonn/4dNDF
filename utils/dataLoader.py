import sys
import os

import torch
import open3d as o3d
import numpy as np

import quaternion
from numpy.linalg import inv

def read_calib_file(filename):
        """ 
            read calibration file (with the kitti format)
            returns -> dict calibration matrices as 4*4 numpy arrays
        """
        calib = {}
        calib_file = open(filename)
        key_num = 0

        for line in calib_file:
            # print(line)
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]
            pose = np.zeros((4,4))
            
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()
        return calib

def read_poses_file(filename, calibration=None):
        """ 
            read pose file (with the kitti format)
        """
        pose_file = open(filename)

        poses = []
        Tr = calibration["Tr"]
        Tr_inv = inv(Tr)

        for line in pose_file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr))) # lidar pose in world frame

        pose_file.close()
        return poses

def read_poses_file_cofusion(filename):
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


class dataLoader():
    def __init__(self, configs):
        self.dataset_name = configs.dataset_name
        self.points_floder = configs.data_path
        self.points_path_list = os.listdir(configs.data_path)
        self.points_path_list.sort(key=lambda x:int((x.split('.')[0])))

        if self.dataset_name == 'kth' :
            translation_file = open(configs.pose_path)
            self.poses = []
            for line in translation_file:
                values = [float(v) for v in line.strip().split()]
                self.poses.append(values)
        elif self.dataset_name == 'newer_college' or self.dataset_name == 'kitti' :
            calib = read_calib_file(configs.calib_path)
            self.poses = read_poses_file(configs.pose_path, calib)
        elif self.dataset_name == 'cofusion' :
            self.poses = read_poses_file_cofusion(configs.pose_path)

        self.device = configs.device
 
    def frame_raw(self, frame_id) -> torch.tensor:
        path = self.points_floder + '/' + self.points_path_list[frame_id]
        np_points = self.read_point_cloud(path)
        torch_points = torch.from_numpy(np_points).to(self.device)
        return torch_points

    def frame_raw_intensity(self, frame_id) ->torch.tensor:

        path = self.points_floder + self.points_path_list[frame_id]
        np_points = self.read_point_cloud(path, intensity=True)
        torch_points = torch.from_numpy(np_points).to(self.device)
        return torch_points

    def frame_transfered(self, frame_id) -> torch.tensor:
        points = self.frame_raw(frame_id)
        if self.dataset_name == 'kth' :
            return points
        else :
            pose = torch.tensor(self.poses[frame_id], device=self.device)
            allones = torch.ones(points.shape[0],1, device=self.device)
            points_homo = torch.cat((points,allones),1).double()
            points_trans = (torch.mm(pose,points_homo.T).T)[:,0:-1]
        return points_trans
    
    def label(self, frame_id) -> torch.tensor:
        
        path = self.labels_floder + self.labels_path_list[frame_id]
        np_labels = self.read_labels(path)
        torch_labels = torch.from_numpy(np_labels).to(self.device)

        return torch_labels
    
    def translation(self, frame_id) -> torch.tensor:
        if self.dataset_name == 'kth' :
            current_translation = torch.tensor(self.poses[frame_id]).to(self.device)
        else :
            pose = torch.tensor(self.poses[frame_id], device=self.device)
            current_translation = pose[0:-1,-1]
        return current_translation
    

    def read_point_cloud(self, filename: str, intensity=False) -> np.ndarray:
        # read point cloud from either (*.ply, *.pcd) or (kitti *.bin) format
        if ".bin" in filename:
            # we also read the intensity channel here
            points = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))
            if not intensity:
                points = points[:,0:3]
        elif ".ply" in filename or ".pcd" in filename:
            pc_load = o3d.io.read_point_cloud(filename)
            points = np.asarray(pc_load.points)
        else:
            sys.exit("The format of the imported point cloud is wrong (support only *pcd, *ply and *bin)")

        return points
    
    def read_labels(self, filename: str) -> np.ndarray:
        if ".label" in filename:
            labels = np.fromfile(filename, dtype=np.uint32).reshape(-1)
        else:
            sys.exit("The format of the imported point labels is wrong (support only *label)")

        labels = labels & 0xFFFF

        labels = np.array(labels, dtype=np.int32)

        return labels
