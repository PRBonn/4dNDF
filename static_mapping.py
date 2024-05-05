import math
import sys
import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

import open3d as o3d
import numpy as np
from tqdm import tqdm

from utils.config import Config
from utils.dataLoader import dataLoader
from utils.dataSampler import dataSampler
from utils.visualizer import MapVisualizer
from utils import tools
from utils import mesher

from model.neural_voxel_hash import NeuralVoxelHash
from model.DCTdecoder import Decoder
from model import sdfloss 

from pytorch3d.ops import knn_points

torch.set_default_dtype(torch.float32)

def static_mapping(configs):
    max_x, min_x = -sys.maxsize, sys.maxsize
    max_y, min_y = -sys.maxsize, sys.maxsize
    max_z, min_z = -sys.maxsize, sys.maxsize

    dataset = dataLoader(configs)
    sampler = dataSampler(configs)
    feature_field = NeuralVoxelHash(configs.feature_dim, \
                                    configs.leaf_voxel_size, \
                                    configs.voxel_level_num, \
                                    configs.scale_up_factor, \
                                    configs.hash_buffer_size, \
                                    configs.device)

    step = configs.step_frame
    if cfg.end_frame == -1 :
        seq_length = math.ceil((len(dataset.poses)-cfg.begin_frame)/step)
    else :
        seq_length = math.ceil(min(cfg.end_frame+1-cfg.begin_frame, len(dataset.poses))/step)

    decoder = Decoder(cfg, seq_length)

    end_points = torch.tensor([], device='cuda', dtype=torch.float32)
    start_points = torch.tensor([], device='cuda', dtype=torch.float32)
    certain_free_points = torch.tensor([], device='cuda', dtype=torch.float32)
    ray_times = torch.tensor([], device='cuda', dtype=torch.long)
 
    with tqdm(total=seq_length) as databar:
        databar.set_description('Collecting Data')
        for i in range(seq_length):
            time_step = i
            frame_idx = i*step + cfg.begin_frame

            frame_points = dataset.frame_transfered(frame_idx)
            frame_translation = dataset.translation(frame_idx)

            points_trans_distances = torch.norm(frame_points - frame_translation, p=2, dim=1, keepdim=False)
            points_trans_mask = (points_trans_distances<configs.valid_radius) 
            frame_points = frame_points[points_trans_mask]
            
            max_x = max([torch.max(frame_points[:,0]).item(), max_x])
            min_x = min([torch.min(frame_points[:,0]).item(), min_x])
            max_y = max([torch.max(frame_points[:,1]).item(), max_y])
            min_y = min([torch.min(frame_points[:,1]).item(), min_y])
            max_z = max([torch.max(frame_points[:,2]).item(), max_z])
            min_z = min([torch.min(frame_points[:,2]).item(), min_z])

            if configs.down_sample:
                down_sampled_id = tools.voxel_down_sample_torch(frame_points, configs.voxel_down_sample_m)
                frame_points_down = frame_points[down_sampled_id]
            else:
                frame_points_down = frame_points

            trans_tensor = frame_translation.repeat(frame_points_down.shape[0],1)
            time_tensor = torch.ones(trans_tensor.shape[0], 1, dtype=ray_times.dtype, device=ray_times.device)*time_step

            surface_sample, _, knn_free_sample, _ = sampler.ray_sample(frame_points_down, trans_tensor, 5)
            feature_field.update(surface_sample)

            sample_distances = torch.norm(knn_free_sample - frame_translation, p=2, dim=1, keepdim=False)

            distances_mask = sample_distances < configs.certain_free_radius
            valid_free_sample = knn_free_sample[distances_mask]

            dists, _, _ = knn_points(p1=valid_free_sample.unsqueeze(0),p2=frame_points_down.unsqueeze(0),K=1,return_nn=False)
            certain_free_mask = (dists > (configs.truncated_length + 0.87*configs.voxel_down_sample_m)).view(-1)

            certain_free_points = torch.cat((certain_free_points, valid_free_sample[certain_free_mask]), dim=0)

            end_points = torch.cat((end_points, frame_points_down), dim=0)

            start_points = torch.cat((start_points, trans_tensor), dim=0)
            ray_times = torch.cat((ray_times, time_tensor),dim=0)
            databar.update(1)

    field_param = list(feature_field.parameters())
    dctmlp_param = list(decoder.parameters())

    field_param_opt_dict = {'params': field_param, 'lr': configs.learning_rate}
    dctmlp_param_opt_dict = {'params': dctmlp_param, 'lr': configs.learning_rate}
    opt = optim.Adam([field_param_opt_dict, dctmlp_param_opt_dict], betas=(0.9,0.99), eps = 1e-15)

    max_step = configs.ekinoal_max_step
    min_step = configs.ekinoal_min_step
    # start training
    with tqdm(total=configs.epochs) as pbar:
        pbar.set_description('traning')
        for epoch in range(configs.epochs):
            random_idx = torch.randperm(end_points.size(0), device='cuda')
        
            this_end_points = end_points[random_idx]
            e_step = max_step - epoch*(max_step-min_step)/configs.epochs

            iterations = math.ceil(this_end_points.shape[0]/configs.batch_size)

            with tqdm(total=iterations) as iter_pbar:
                for i in range(iterations):
                    start_idx = configs.batch_size*i
                    end_idx = min(configs.batch_size*(i+1), random_idx.shape[0])

                    iter_indices = random_idx[start_idx:end_idx]

                    # print(iter_indices)
                    iter_end_points = end_points[iter_indices]
                    iter_start_points = start_points[iter_indices]
                    iter_ray_times = ray_times[iter_indices].reshape(-1,1)

                    surface_sample, surface_pd, free_samples, _ = sampler.ray_sample(iter_end_points, iter_start_points, configs.free_sample_num)
                    
                    surface_time = iter_ray_times.repeat(1, configs.truncated_sample_num+configs.occupied_sample_num).reshape(-1,1)
                    surface_features = feature_field.get_features(surface_sample, surface_time.long())
                    surface_static_pred, surface_dynamic_pred, surface_weights = decoder(surface_features, surface_time.long())

                    surface_loss = sdfloss.sdfLoss(surface_dynamic_pred, surface_pd).mean()
                    
                    free_time = iter_ray_times.repeat(1,  configs.free_sample_num).reshape(-1,1)
                    free_features = feature_field.get_features(free_samples, free_time.long())
                    free_static_pred, free_dynamic_pred, _ = decoder(free_features, free_time.long())

                    tloss = torch.abs(free_dynamic_pred - configs.truncated_length).mean()
                    
                    eikonal_indices = torch.randint(0, surface_sample.shape[0], (iter_end_points.shape[0],), device='cuda')
                    eikonal_sample = surface_sample[eikonal_indices]
                    eikonal_times = surface_time[eikonal_indices]

                    _, d_normal, _ = sdfloss.double_numerical_normals(feature_field, decoder, eikonal_sample, eikonal_times, e_step)

                    eikonal_loss = torch.abs(d_normal.norm(2,dim=-1) - 1.0)

                    certain_free_indices = torch.randint(0, certain_free_points.shape[0], (configs.batch_size, ), device='cuda')
                    certain_free_samples = certain_free_points[certain_free_indices]
                    certain_free_features = feature_field.get_features(certain_free_samples)
                    certain_time = torch.zeros(certain_free_samples.shape[0], dtype=torch.int64 , device='cuda')
                    certain_static_pred, _, _ = decoder(certain_free_features, certain_time.long())

                    certain_free_loss = torch.abs(certain_static_pred - configs.truncated_length).mean()

                    loss = surface_loss + configs.ekional_lamda*eikonal_loss.mean() + \
                                          configs.free_space_lamda*tloss + \
                                          configs.certain_free_lamda*certain_free_loss

                    loss.backward()
                    opt.step()
                    opt.zero_grad()

                    iter_pbar.update(1)
            pbar.update(1)

    if configs.mesh_recon:
        print('marching cubes ...')
        if configs.mesh_dynamic:
            for i in range(seq_length):
                output_path_pertime = configs.output_folder + '/mesh_' + str(i)
                mesher.create_mesh(feature_field, decoder, i, output_path_pertime, max_x, min_x, max_y, min_y, max_z, min_z, configs.mesh_resolution, static=False, scale=1)
        else :
            output_path = configs.output_folder + '/mesh'
            mesher.create_mesh(feature_field, decoder, 0, output_path, max_x, min_x, max_y, min_y, max_z, min_z, configs.mesh_resolution, static=True, scale=1)
    
    if configs.static_pointcloud:
        threshold = configs.segmentation_threshold
        if configs.point_cloud_viewer: 
            vis = MapVisualizer()
        total_static_ps = torch.tensor([], device='cuda')
        for i in range(seq_length):
            time_step = i
            frame_idx = i*step+cfg.begin_frame
            current_scan = dataset.frame_transfered(frame_idx)
            current_translation = dataset.translation(frame_idx)

            current_trans_distances = torch.norm(current_scan - current_translation, p=2, dim=1, keepdim=False)
            current_trans_mask = (current_trans_distances<configs.valid_radius) 
            current_scan = current_scan[current_trans_mask]

            scan_features = feature_field.get_features(current_scan.contiguous())
            scan_t = torch.ones(current_scan.shape[0], dtype=torch.int64 , device='cuda')*time_step
            static_sdf, _, _ = decoder(scan_features, scan_t.long())
            
            mask = (static_sdf > threshold)

            dynamic_ps_torch = current_scan[mask].detach()

            dynamic_ps = o3d.geometry.PointCloud()
            dynamic_ps.points = o3d.utility.Vector3dVector(dynamic_ps_torch.cpu().numpy())

            static_ps_torch = current_scan[~mask].detach()
            static_ps = o3d.geometry.PointCloud()
            static_ps.points = o3d.utility.Vector3dVector(static_ps_torch.cpu().numpy())

            total_static_ps = torch.cat((total_static_ps, static_ps_torch),0)

            # comment out to store the dynamic and static points
            # o3d.io.write_point_cloud(configs.output_folder+'/static_' + str(i) + '.ply', static_ps)
            # o3d.io.write_point_cloud(configs.output_folder+'/dynamic_' + str(i) + '.ply', dynamic_ps)
            
            if configs.point_cloud_viewer:
                vis.update(scan=static_ps, dynamic_points=dynamic_ps)
                time.sleep(0.2)

        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float32
        static_map = o3d.t.geometry.PointCloud(device)
        static_map.point['positions'] = o3d.core.Tensor(total_static_ps.cpu().numpy(), dtype, device)
        output_path = configs.output_folder + '/static_points.pcd'
        o3d.t.io.write_point_cloud(output_path, static_map, print_progress=False)


if __name__ == "__main__":
    cfg = Config()

    if len(sys.argv) > 1:
        cfg.load(sys.argv[1])
    else:
        sys.exit("No config file.")
    
    if not os.path.exists(cfg.output_folder):
        os.makedirs(cfg.output_folder)

    tools.seed_everything(cfg.random_seed)

    static_mapping(cfg)







