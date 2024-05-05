import os
import sys
import torch

import numpy as np
import quaternion
import open3d as o3d
from numpy.linalg import inv
from pykdtree.kdtree import KDTree

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

def read_point_cloud(filename: str, color_on: bool = False) -> np.ndarray:
    # read point cloud from either (*.ply, *.pcd) or (kitti *.bin) format
    if ".bin" in filename:
        # we also read the intensity channel here
        points = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))
        points = points[:,0:3]
    elif ".ply" in filename or ".pcd" in filename:
        pc_load = o3d.io.read_point_cloud(filename)
        points = np.asarray(pc_load.points)
        if pc_load.has_colors() and color_on:           
            colors = np.asarray(pc_load.colors) # if they are available
            points = np.hstack((points, colors))
    else:
        sys.exit("The format of the imported point cloud is wrong (support only *pcd, *ply and *bin)")

    return points # as np


def voxel_down_sample_torch(points: torch.tensor, voxel_size: float):
    """
        voxel based downsampling. Returns the indices of the points which are closest to the voxel centers. 
    Args:
        points (torch.Tensor): [N,3] point coordinates
        voxel_size (float): grid resolution

    Returns:
        indices (torch.Tensor): [M] indices of the original point cloud, downsampled point cloud would be `points[indices]`  

    Reference: Louis Wiesmann
    """
    _quantization = 1000

    offset = torch.floor(points.min(dim=0)[0]/voxel_size).long()
    grid = torch.floor(points / voxel_size)
    center = (grid + 0.5) * voxel_size
    dist = ((points - center) ** 2).sum(dim=1)**0.5
    dist = dist / dist.max() * (_quantization - 1) # for speed up

    grid = grid.long() - offset
    v_size = grid.max().ceil()
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size

    unique, inverse = torch.unique(grid_idx, return_inverse=True)
    idx_d = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
       
    offset = 10**len(str(idx_d.max().item()))

    idx_d = idx_d + dist.long() * offset
    idx = torch.empty(unique.shape, dtype=inverse.dtype,
                      device=inverse.device).scatter_reduce_(dim=0, index=inverse, src=idx_d, reduce="amin", include_self=False)
    idx = idx % offset
    return idx

def distance_p2p(points_src, points_tgt):
    kdtree = KDTree(points_tgt)
    dist, _ = kdtree.query(points_src)
    return dist

def get_threshold_percentage(dist, threshold):
    in_threshold = (dist <= threshold).mean()
    return in_threshold


class MeshEvaluator:

    def __init__(self, threshold):
        self.thresholds = threshold

    def _evaluate(self, pointcloud, pointcloud_tgt):

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completenes = distance_p2p(pointcloud_tgt, pointcloud)
        recall = get_threshold_percentage(completenes, self.thresholds)
        completenes = completenes.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy = distance_p2p(pointcloud, pointcloud_tgt)
        precision = get_threshold_percentage(accuracy, self.thresholds)
        accuracy = accuracy.mean()

        # Chamfer distance
        chamfer_l1 = 0.5 * (completenes + accuracy)

        # F-Score
        F = 2 * precision * recall / (precision + recall)
        
        # com_str = ("completenes: %d" %(completenes))
        # print("completenes: ", completenes)
        # print("accuracy: ", accuracy)
        # print("chamfer_l1: ", chamfer_l1)
        # print("F-score: ", F)


        return {"completeness": completenes, "accuracy": accuracy, "chamfer_l1": chamfer_l1, "f_score": 100 * F}