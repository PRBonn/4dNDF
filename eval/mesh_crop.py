import torch
import open3d as o3d
import numpy as np
import neural_voxel_hash_simple

NAN_METRIC = float('nan')

reference_ply = '../data/newer_college/reference_newer_college.ply'
est_ply = '../output/newer_college/mesh.ply'

o3d.utility.random.seed(0)

if __name__ == "__main__":

    mask_field = neural_voxel_hash_simple.NeuralVoxelHashSimple(1, 0.26, 1, 1.5, int(5e7), device='cuda')

    reference_mesh = o3d.io.read_triangle_mesh(reference_ply)
    reference_vertices = torch.from_numpy(np.asarray(reference_mesh.vertices)).to('cuda')
    ref_sampled_pcd = reference_mesh.sample_points_uniformly(number_of_points=reference_vertices.shape[0], use_triangle_normal=False)
    ref_sampled_points = torch.tensor(np.asarray(ref_sampled_pcd.points), device='cuda')
    mask_field.update(ref_sampled_points)

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

    o3d.io.write_triangle_mesh('../output/newer_college/croped_mesh.ply', mesh)
