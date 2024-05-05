import torch
import torch.nn as nn

class NeuralVoxelHashSimple():

    def __init__(self, feature_dim, leaf_voxel_size, voxel_level_num, scale_up_factor, hash_buffer_size, device) -> None:
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

        # self.to(self.device)

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
    
    def to_corners(self, points: torch.Tensor, resolution):
        origin_corner = torch.floor(points / resolution)
        corners = (origin_corner.repeat(1,8) + self.steps.reshape(1,-1)).reshape(-1,3)
        return corners

    def get_valid_mask(self, query_points):
        n = self.voxel_level_num-1
        current_resolution = self.leaf_voxel_size*(self.scale_up_factor**n)
        
        query_corners = self.to_corners(query_points, current_resolution).to(self.primes)
 
        query_keys = (query_corners * self.primes).sum(-1) % self.buffer_size

        hash_index_nx8 = self.feature_indexs_list[n][query_keys].reshape(-1,8)

        featured_query_mask = (hash_index_nx8.min(dim=1)[0]) > -1

        return featured_query_mask
