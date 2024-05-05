import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import grad

import open3d as o3d
import numpy as np

import os

import eval_utils

import neural_voxel_hash_simple

NAN_METRIC = float('nan')

gt_data_dir = '../data/cofusion/ply_static/'
input_data_dir = '../data/cofusion/ply_noise/'
camera_poses = '../data/cofusion/gt-cam-0.txt'

# mesh need to be evaluated
est_ply = '../output/cofusion/mesh.ply'

# baselines
# est_ply = '../data/baseline/cofusion/nksr.ply'
# est_ply = '../data/baseline/cofusion/shine.ply'
# est_ply = '../data/baseline/cofusion/vdb_fusion.ply'

step = 2
down_sampled_size = 0.005

mask_field = neural_voxel_hash_simple.NeuralVoxelHashSimple(1, down_sampled_size, 1, 1.5, int(4e7), device='cuda')

if __name__ == "__main__":

    path_list = os.listdir(gt_data_dir)
    path_list.sort(key=lambda x:int((x.split('.')[0])))

    input_path_list = os.listdir(input_data_dir)
    input_path_list.sort(key=lambda x:int((x.split('.')[0])))
    poses = eval_utils.readPosesFile(camera_poses)

    seq_length = len(poses)
    gt_points = torch.tensor([], device='cuda', dtype=torch.float32)
    input_demo_points = torch.tensor([], device='cuda', dtype=torch.float32)
    for i in range(0, seq_length, step):
        path = gt_data_dir+path_list[i]
        np_points = eval_utils.read_point_cloud(path)

        points = torch.tensor(np_points, device='cuda')
        pose = torch.tensor(poses[i], device='cuda')
        allones = torch.ones(points.shape[0],1).cuda()
        points_homo = torch.cat((points,allones),1)
        points_trans = (torch.mm(pose,points_homo.T).T)[:,0:-1]

        gt_points = torch.cat((gt_points, points_trans), dim=0)

        down_sampled_id = eval_utils.voxel_down_sample_torch(gt_points, down_sampled_size)
        gt_points = gt_points[down_sampled_id]

        input_path = input_data_dir+input_path_list[i]
        input_np_points = eval_utils.read_point_cloud(input_path)
        
        input_points = torch.tensor(input_np_points, device='cuda')
        
        allones = torch.ones(input_points.shape[0],1).cuda()
        input_points_homo = torch.cat((input_points,allones),1)
        input_points_trans = (torch.mm(pose,input_points_homo.T).T)[:,0:-1]

        mask_field.update(input_points_trans)


    mesh = o3d.io.read_triangle_mesh(est_ply)
    vertices = torch.from_numpy(np.asarray(mesh.vertices)).to('cuda')
    index = torch.from_numpy(np.asarray(mesh.triangles)).to('cuda').long()
    
    bool_mask = torch.zeros(index.shape[0]*3,1,device=index.device)
    tri_vertices = vertices[index.reshape(-1,1)].squeeze(1)

    valid_mask = mask_field.get_valid_mask(tri_vertices)
    bool_mask[valid_mask] = 1

    non_mask = bool_mask.reshape(-1,3).sum(1) < 2.5
    valid_index = index[~non_mask]

    mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices.cpu().numpy()),
            o3d.utility.Vector3iVector(valid_index.cpu().numpy())
    )

    o3d.utility.random.seed(0)
    sampled_pcd = mesh.sample_points_uniformly(number_of_points=gt_points.shape[0], use_triangle_normal=False)
    sampled_points = torch.tensor(np.asarray(sampled_pcd.points), device='cuda').cpu().numpy()

    evaluator = eval_utils.MeshEvaluator(0.01)
    results = evaluator._evaluate(sampled_points, gt_points.cpu().numpy())

            # com_str = ("completenes: %d" %(completenes))
    ## we report the accurracy and Champfer distance in cm:
    print(f"completeness: {100 * results['completeness']:.3f}")
    print(f"accuracy:     {100 * results['accuracy']:.3f}")
    print(f"chamfer_l1:   {100 * results['chamfer_l1']:.3f}")
    print(f"F-score:      {results['f_score']:.2f}")
