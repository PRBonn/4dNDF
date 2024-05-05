import torch
import torch.nn as nn

import math
import time
import sys
from tqdm import tqdm
import open3d as o3d

import numpy as np

class NeuralVoxelHash(nn.Module):

    def __init__(self, feature_dim, leaf_voxel_size, voxel_level_num, scale_up_factor, hash_buffer_size, device) -> None:
        
        super().__init__()
        # feature setting
        self.feature_dim = feature_dim
        self.feature_std = 0.01

        # map structure setting
        self.leaf_voxel_size = leaf_voxel_size
        self.voxel_level_num = voxel_level_num
        self.scale_up_factor = scale_up_factor

        self.dtype = torch.float32
        self.device = device

        self.buffer_size = hash_buffer_size

        # hash function
        self.primes = torch.tensor(
            [73856093, 19349669, 83492791], dtype=torch.int64, device=self.device)
        # for point to corners
        self.steps = torch.tensor([[0., 0., 0.], [0., 0., 1.], 
                                   [0., 1., 0.], [0., 1., 1.], 
                                   [1., 0., 0.], [1., 0., 1.], 
                                   [1., 1., 0.], [1., 1., 1.]], dtype=self.dtype, device=self.device)
        
        self.features_list = nn.ParameterList([])
        self.feature_time_steps = []
        self.feature_indexs_list = []
        # self.corner_list = []
        for l in range(self.voxel_level_num):
            features = nn.Parameter(torch.tensor([],device=self.device))
            time_steps = torch.tensor([], dtype=torch.int64, device=self.device)
            feature_indexs = torch.full([self.buffer_size], -1, dtype=torch.int64, 
                                             device=self.device) # -1 for not valid (occupied)
            self.features_list.append(features)
            self.feature_time_steps.append(time_steps)
            self.feature_indexs_list.append(feature_indexs)

        self.to(self.device)

    def update(self, points: torch.Tensor):
        for i in range(self.voxel_level_num):
            current_resolution = self.leaf_voxel_size*(self.scale_up_factor**i)

            corners = self.to_corners(points, current_resolution)

            # remove reptitive coordinates
            offset = corners.min(dim=0,keepdim=True)[0]
            shift_corners = corners - offset
            v_size = shift_corners.max() + 1
            corner_idx = shift_corners[:, 0] + shift_corners[:, 1] * v_size + shift_corners[:, 2] * v_size * v_size
            unique, index, counts = torch.unique(corner_idx, sorted=False ,return_inverse=True, return_counts=True)

            # unique_corner_z = torch.trunc(unique / (v_size * v_size))
            # zdifference = unique_corner_z * v_size * v_size
            # unique_corner_y = torch.trunc((unique - zdifference) / v_size)
            # unique_corner_x = unique - zdifference - unique_corner_y * v_size
            # unique_corners = torch.stack((unique_corner_x, unique_corner_y, unique_corner_z), dim=0).T.to(self.primes) + offset

            unique_corners = torch.zeros_like(corners[:len(counts), :], device=corners.device)
            index.unsqueeze_(-1)
            unique_corners.scatter_add_(-2, index.expand(corners.shape), corners)
            unique_corners /= counts.unsqueeze(-1)

            # hash function
            keys = (unique_corners.to(self.primes) * self.primes).sum(-1) % self.buffer_size

            update_mask = (self.feature_indexs_list[i][keys] == -1)

            new_feature_count = unique_corners[update_mask].shape[0]

            self.feature_indexs_list[i][keys[update_mask]] = torch.arange(new_feature_count, dtype=self.feature_indexs_list[i].dtype, 
                                                                        device=self.feature_indexs_list[i].device) + self.features_list[i].shape[0]
            
            new_fts = self.feature_std*torch.randn(new_feature_count, self.feature_dim, device=self.device, dtype=self.dtype)


            new_time_step = torch.ones(new_feature_count, 2, dtype=torch.int64, device=self.device)
            new_time_step[:,0] *= self.buffer_size
            new_time_step[:,1] *= -1
            self.features_list[i] = nn.Parameter(torch.cat((self.features_list[i], new_fts),0))
            self.feature_time_steps[i] = torch.cat((self.feature_time_steps[i], new_time_step),0)



    def get_features(self, query_points, t=None): 
        sum_features = torch.zeros(query_points.shape[0], self.feature_dim, device=self.device, dtype=self.dtype)
        # valid_mask = torch.ones(query_points.shape[0], device=self.device, dtype=bool)
        for i in range(self.voxel_level_num):
           current_resolution = self.leaf_voxel_size*(self.scale_up_factor**i)

           query_corners = self.to_corners(query_points, current_resolution).to(self.primes)
 
           query_keys = (query_corners * self.primes).sum(-1) % self.buffer_size

           hash_index_nx8 = self.feature_indexs_list[i][query_keys].reshape(-1,8)

           featured_query_mask = (hash_index_nx8.min(dim=1)[0]) > -1

           features_index = hash_index_nx8[featured_query_mask].reshape(-1,1).squeeze(1)

           # update time stamps for every feature vector
           if t is not None:
               valid_t = ((t[featured_query_mask].reshape(-1,1)).repeat(1,8)).reshape(-1,1).squeeze(1)
               self.feature_time_steps[i][:,0].scatter_reduce_(0, features_index, valid_t, reduce="amin")
               self.feature_time_steps[i][:,1].scatter_reduce_(0, features_index, valid_t, reduce="amax")

           coeffs = self.interpolat(query_points[featured_query_mask], current_resolution)
           
           sum_features[featured_query_mask] += (self.features_list[i][features_index]*coeffs).reshape(-1,8,self.feature_dim).sum(1)

        # sum_features = torch.cat([sum_features, query_points], 1)

        return sum_features#, valid_mask
    
    # only for leaf nodes
    def get_time_interval(self, query_points):

        output = torch.zeros(query_points.shape[0], 2, dtype=torch.int64, device=query_points.device) 
        
        query_corners = self.to_corners(query_points, self.leaf_voxel_size).to(self.primes)
 
        query_keys = (query_corners * self.primes).sum(-1) % self.buffer_size

        hash_index_nx8 = self.feature_indexs_list[0][query_keys].reshape(-1,8)

        featured_query_mask = (hash_index_nx8.min(dim=1)[0]) > -1

        features_index = hash_index_nx8[featured_query_mask].reshape(-1,1).squeeze(1)

        start_indices,_ = torch.max((self.feature_time_steps[0][features_index].reshape(-1,8,2))[:,:,0],dim=1)
        end_indices,_ = torch.min((self.feature_time_steps[0][features_index].reshape(-1,8,2))[:,:,1],dim=1)

        valid_interval = torch.stack((start_indices,end_indices),0).T
        output[featured_query_mask] = valid_interval

        return output
    

    def get_valid_mask(self, query_points):
        n = self.voxel_level_num-1
        current_resolution = self.leaf_voxel_size*(self.scale_up_factor**n)
        
        query_corners = self.to_corners(query_points, current_resolution).to(self.primes)
 
        query_keys = (query_corners * self.primes).sum(-1) % self.buffer_size

        hash_index_nx8 = self.feature_indexs_list[n][query_keys].reshape(-1,8)

        featured_query_mask = (hash_index_nx8.min(dim=1)[0]) > -1

        return featured_query_mask

    
    def interpolat(self, x, resolution):
        coords = x / resolution
        d_coords = coords - torch.floor(coords)
        tx = d_coords[:,0]
        _1_tx = 1-tx
        ty = d_coords[:,1]
        _1_ty = 1-ty
        tz = d_coords[:,2]
        _1_tz = 1-tz
        p0 = _1_tx*_1_ty*_1_tz
        p1 = _1_tx*_1_ty*tz
        p2 = _1_tx*ty*_1_tz
        p3 = _1_tx*ty*tz
        p4 = tx*_1_ty*_1_tz
        p5 = tx*_1_ty*tz
        p6 = tx*ty*_1_tz
        p7 = tx*ty*tz
        p = torch.stack((p0,p1,p2,p3,p4,p5,p6,p7),0).T.reshape(-1,1)
        return p

    def to_corners(self, points: torch.Tensor, resolution):
        origin_corner = torch.floor(points / resolution)
        corners = (origin_corner.repeat(1,8) + self.steps.reshape(1,-1)).reshape(-1,3)
        return corners
    
    def to_lines(self, points: torch.Tensor, resolution):

        middles = torch.tensor([[0., 0., 0.5], 
                                [0., 0.5, 0.], 
                                [0.5, 0., 0.], 
                                [0.5, 0., 1.], 
                                [0., 0.5, 1.],
                                [1., 0.5, 1.],
                                [1., 0., 0.5],
                                [0.5, 1., 1.],
                                [1., 1., 0.5],
                                [1., 0.5, 0.],
                                [0.5, 1., 0.],
                                [0., 1., 0.5]], dtype=points.dtype, device=points.device)
        
        rots = torch.tensor([[0., 0., 0.], 
                             [0.5 * np.pi, 0., 0.], 
                             [0., 0.5 * np.pi, 0.],
                             [0., 0.5 * np.pi, 0.], 
                             [0.5 * np.pi, 0., 0.],
                             [0.5 * np.pi, 0., 0.],
                             [0., 0., 0.],
                             [0., 0.5 * np.pi, 0.],
                             [0., 0., 0.],
                             [0.5 * np.pi, 0., 0.],
                             [0., 0.5 * np.pi, 0.],
                             [0., 0., 0.]], dtype=points.dtype, device=points.device)
        
        corners = torch.floor(points / resolution)

        offset = corners.min(dim=0,keepdim=True)[0]
        shift_corners = corners - offset
        v_size = shift_corners.max() + 1
        corner_idx = shift_corners[:, 0] + shift_corners[:, 1] * v_size + shift_corners[:, 2] * v_size * v_size
        unique, index, counts = torch.unique(corner_idx, sorted=False ,return_inverse=True, return_counts=True)


        unique_corners = torch.zeros_like(corners[:len(counts), :], device=corners.device)
        index.unsqueeze_(-1)
        unique_corners.scatter_add_(-2, index.expand(corners.shape), corners)
        unique_corners /= counts.unsqueeze(-1)

        corners = (unique_corners.repeat(1,12) + middles.reshape(1,-1)).reshape(-1,3)
        rotations = rots.repeat(unique_corners.shape[0],1)

        return corners*resolution, rotations

