#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
import math
import open3d as o3d
import pdb


def create_mesh(
   field,
   decoder,
   t, 
   filename,
   max_x,
   min_x,
   max_y,
   min_y, 
   max_z,
   min_z,
   voxel_scale,
   static=False, 
   max_batch=8192, 
   offset=None, 
   scale=None
):
    start = time.time()
    ply_filename = filename + '.ply'

    length_x = max_x - min_x
    length_y = max_y - min_y
    length_z = max_z - min_z
    voxel_num_x = math.ceil(length_x/voxel_scale)+1
    voxel_num_y = math.ceil(length_y/voxel_scale)+1
    voxel_num_z = math.ceil(length_z/voxel_scale)+1

    total_sample_num = voxel_num_x*voxel_num_y*voxel_num_z
    samples = torch.zeros(total_sample_num, 4)

    voxel_origin = [min_x-voxel_scale, min_y-voxel_scale, min_z-voxel_scale]

    samples.requires_grad = False

    x = torch.arange(voxel_num_x, dtype=torch.long)
    y = torch.arange(voxel_num_y, dtype=torch.long)
    z = torch.arange(voxel_num_z, dtype=torch.long)
    
    x, y, z = torch.meshgrid(x, y, z, indexing='ij')

    coord = torch.stack((x.flatten(), y.flatten(), z.flatten()))

    samples = torch.zeros(total_sample_num, 5) #x,y,z,pred,mask
    samples[:, 0:3] = torch.transpose(coord, 0, 1)
    #print(samples)

    # transform first 3 columns to be the x, y, z coordinate
    samples[:, 0:3] *= voxel_scale
    samples[:, 0] += voxel_origin[0]
    samples[:, 1] += voxel_origin[1]
    samples[:, 2] += voxel_origin[2]
    
    head = 0
    while head < total_sample_num:
        sample_subset = samples[head : min(head + max_batch, total_sample_num), 0:3].cuda()
        # input_indice = field.getIndices(sample_subset)     
        # features,_ = field.getFeatures(input_indice, sample_subset)
        # features , _ = field.get_features(sample_subset)
        features = field.get_features(sample_subset)
        time_vector = (torch.ones(1, sample_subset.shape[0], device='cuda')*t).squeeze(0).long()
        if static:
            # time_intervals = field.get_time_interval(sample_subset.contiguous())
            # output, _ = decoder.interval_max(features.float(), time_intervals)
            output,_,_ = decoder(features.float(), time_vector)
        else:
            _,output,_ = decoder(features.float(), time_vector)
        samples[head : min(head + max_batch, total_sample_num), 3] = (
            output
            .detach()
            .cpu()
        )

        head += max_batch
    
    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(voxel_num_x, voxel_num_y, voxel_num_z).numpy()

    mesh = convert_sdf_samples_to_ply(
                            sdf_values,
                            voxel_origin,
                            voxel_scale,
                            ply_filename,
                            offset,
                            scale,)

    end = time.time()
    print("meshing takes: %f" % (end - start))
    return mesh

def convert_sdf_samples_to_ply(
    sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    file_path,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    verts, faces, normals, values = skimage.measure.marching_cubes(
        sdf_tensor, level=0.0, spacing=[voxel_size] * 3)

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points * scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    # num_verts = verts.shape[0]
    # num_faces = faces.shape[0]

    # verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    # for i in range(0, num_verts):
    #     verts_tuple[i] = tuple(mesh_points[i, :])

    # faces_building = []
    # for i in range(0, num_faces):
    #     faces_building.append(((faces[i, :].tolist(),)))
    # faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh_points),
        o3d.utility.Vector3iVector(faces)
    )

    o3d.io.write_triangle_mesh(file_path, mesh)
    return mesh
    