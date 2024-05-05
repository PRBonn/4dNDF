import yaml
import os
import torch

class Config:
    def __init__(self):

        # Default values
        self.dataset_name: str = ""
        self.data_path: str = ""  # input point cloud folder
        self.pose_path: str = ""  # input pose file
        self.calib_path: str = ""  # input calib file (to sensor frame)
        self.label_path: str = "" # input point-wise label path
        self.output_folder: str = ""  # output root folder
        self.random_seed: int = 0

        self.begin_frame: int = 0  # begin from this frame
        self.end_frame: int = 200  # end at this frame
        self.step_frame: int = 1 

        self.device: str = "cuda"  # use "cuda" or "cpu"
        self.gpu_id: str = "0"  # used GPU id
        self.dtype = torch.float32 # default torch tensor data type

        #preprocessing
        self.valid_radius: float = 200

        # neural voxel hash
        self.leaf_voxel_size: float = 0.3
        self.voxel_level_num : int = 2
        self.scale_up_factor : float = 1.5
        self.hash_buffer_size: int = int(2e7)

        self.feature_dim: int = 8  # length of the feature for each grid feature
        self.feature_std: float = 0.0  # grid feature initialization standard deviation

        # decoder
        self.mlp_level: int = 2
        self.mlp_hidden_dim: int = 64
        self.mlp_basis_num: int = 32

        # sampling
        self.truncated_length: float = 0.5
        self.truncated_sample_num: int = 3
        self.occupied_length: float = 0.5
        self.occupied_sample_num: int = 2
        self.free_sample_num: int = 15
        self.certain_free_radius: int = 15
        self.down_sample: bool = True
        self.voxel_down_sample_m: float = 0.1

        # loss
        self.ekinoal_max_step: float = 0.08
        self.ekinoal_min_step: float = 0.03
        self.ekional_lamda: float = 0.02
        self.free_space_lamda: float = 0.25
        self.certain_free_lamda: float = 0.2

        # mapping
        self.epochs: int = 20
        self.batch_size: int = 16384
        self.learning_rate: float = 0.01

        # output
        self.segmentation_threshold: float = 0.16
        self.static_pointcloud: bool = False
        self.point_cloud_viewer: bool = False
        self.mesh_recon: bool = False
        self.mesh_dynamic: bool = False
        self.mesh_resolution: float = 0.2



    def load(self, config_file):
        config_args = yaml.safe_load(open(os.path.abspath(config_file)))

        # setting
        self.dataset_name = config_args["setting"]["dataset"]
        self.data_path = config_args["setting"]["data_path"] 
        self.pose_path = config_args["setting"]["pose_path"]
        self.calib_path = config_args["setting"]["calib_path"]
        self.label_path = config_args["setting"]["label_path"]
        self.output_folder = config_args["setting"]["output_folder"]  
        self.begin_frame = config_args["setting"]["begin_frame"]
        self.step_frame = config_args["setting"]["step_frame"]
        self.end_frame = config_args["setting"]["end_frame"]
        self.random_seed = config_args["setting"]["random_seed"]
        self.device = config_args["setting"]["device"]
        self.valid_radius = config_args["setting"]["valid_radius"]
  
        # neuralvoxel 
        self.leaf_voxel_size = config_args["neuralvoxel"]["leaf_voxel_size"]
        self.voxel_level_num = config_args["neuralvoxel"]["voxel_level_num"]
        self.scale_up_factor = config_args["neuralvoxel"]["scale_up_factor"]
        self.feature_dim = config_args["neuralvoxel"]["feature_dim"]

        # decoder
        self.mlp_level = config_args["decoder"]["mlp_level"]
        self.mlp_hidden_dim = config_args["decoder"]["mlp_hidden_dim"]
        self.mlp_basis_num = config_args["decoder"]["mlp_basis_num"]

        # sampling
        self.truncated_length = config_args["sampler"]["truncated_length"]
        self.surface_sample_num = config_args["sampler"]["truncated_sample_num"]
        self.occupied_length = config_args["sampler"]["occupied_length"]
        self.occupied_sample_num = config_args["sampler"]["occupied_sample_num"]
        self.free_sample_num = config_args["sampler"]["free_sample_num"]
        self.certain_free_radius = config_args["sampler"]["certain_free_radius"]
        self.down_sample = config_args["sampler"]["down_sample"]
        self.voxel_down_sample_m = config_args["sampler"]["voxel_down_sample_m"]           

        # loss
        self.ekinoal_max_step = config_args["loss"]["ekinoal_max_step"]   
        self.ekinoal_min_step = config_args["loss"]["ekinoal_min_step"]
        self.ekional_lamda = config_args["loss"]["ekional_lamda"]    
        self.free_space_lamda = config_args["loss"]["free_space_lamda"]
        self.certain_free_lamda = config_args["loss"]["certain_free_lamda"]

        # mapping
        self.epochs = config_args["mapping"]["epochs"] 
        self.batch_size = config_args["mapping"]["batch_size"] 
        self.learning_rate = config_args["mapping"]["learning_rate"]

        # output
        self.segmentation_threshold = config_args["output"]["segmentation_threshold"]
        self.static_pointcloud = config_args["output"]["static_pointcloud"]
        self.point_cloud_viewer =config_args["output"]["point_cloud_viewer"]
        self.mesh_recon = config_args["output"]["mesh_recon"]
        self.mesh_dynamic = config_args["output"]["mesh_dynamic"]
        self.mesh_resolution = config_args["output"]["mesh_resolution"]
