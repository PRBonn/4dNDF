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

data_dir = "../data/newer_college/pcd/"
calib_path = "../data/newer_college/calib.txt"
poses_file = "../data/newer_college/poses.txt"

ground_truth_ply = '../data/newer_college/ncd_quad_gt_pc.ply'
est_ply = '../output/newer_college/croped_mesh.ply'

# baselines
# est_ply = '../data/baseline/newer_college/nksr.ply'
# est_ply = '../data/baseline/newer_college/shine.ply'
# est_ply = '../data/baseline/newer_college/vdb_fusion.ply'

mask_field = neural_voxel_hash_simple.NeuralVoxelHashSimple(1, 0.1, 1, 1.5, int(4e7), device='cuda')

if __name__ == "__main__":

    path_list = os.listdir(data_dir)
    path_list.sort(key=lambda x:int((x.split('.')[0])))
    calib = eval_utils.read_calib_file(calib_path)
    poses = eval_utils.read_poses_file(poses_file, calib)
    seq_length = len(poses)
    for i in range(seq_length):
        path = data_dir+path_list[i]
        np_points = eval_utils.read_point_cloud(path)

        points = torch.tensor(np_points, device='cuda')
        pose = torch.tensor(poses[i], device='cuda')
        allones = torch.ones(points.shape[0],1).cuda()
        points_homo = torch.cat((points,allones),1)
        points_trans = (torch.mm(pose,points_homo.T).T)[:,0:-1]
        mask_field.update(points_trans)

    mesh = o3d.io.read_triangle_mesh(est_ply)
    vertices = torch.from_numpy(np.asarray(mesh.vertices)).to('cuda')
    index = torch.from_numpy(np.asarray(mesh.triangles)).to('cuda').long()
    
    bool_mask = torch.zeros(index.shape[0]*3,1,device=index.device)
    tri_vertices = vertices[index.reshape(-1,1)].squeeze(1)

    # Only the observed area is left.
    valid_mask = mask_field.get_valid_mask(tri_vertices)
    bool_mask[valid_mask] = 1

    non_mask = bool_mask.reshape(-1,3).sum(1) < 2.5
    valid_index = index[~non_mask]

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices.cpu().numpy()),
        o3d.utility.Vector3iVector(valid_index.cpu().numpy())
    )

    gt_points = o3d.io.read_point_cloud(ground_truth_ply)
    gt_points_np = np.asarray(gt_points.points)

    # Use the same mask to filter gt_pointcloud
    gt_points_torch = torch.from_numpy(gt_points_np).to('cuda')
    gt_valid_mask = mask_field.get_valid_mask(gt_points_torch)
    gt_valid_points = gt_points_torch[gt_valid_mask]

    # Sample as many points as filtered gt
    o3d.utility.random.seed(0)
    sampled_pcd = mesh.sample_points_uniformly(number_of_points=gt_valid_points.shape[0], use_triangle_normal=False)
    sampled_points = torch.tensor(np.asarray(sampled_pcd.points), device='cuda').cpu().numpy()

    evaluator = eval_utils.MeshEvaluator(0.2)
    results = evaluator._evaluate(sampled_points, gt_valid_points.cpu().numpy())

    ## We report the accurracy and Champfer distance in cm:
    print(f"completeness: {100 * results['completeness']:.3f}")
    print(f"accuracy:     {100 * results['accuracy']:.3f}")
    print(f"chamfer_l1:   {100 * results['chamfer_l1']:.3f}")
    print(f"F-score:      {results['f_score']:.2f}")
