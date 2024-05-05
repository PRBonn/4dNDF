<p align="center">
  <h1 align="center">3D LiDAR Mapping in Dynamic Environments using a 4D Implicit Neural Representation</h1>

  <p align="center">
    <a href="https://github.com/PRBonn/4dNDF"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
    <a href="https://github.com/PRBonn/4dNDF"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/zhong2024cvpr.pdf"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a>
    <a href="https://github.com/PRBonn/4dNDF/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square
  </p>


  <p align="center">
    <a href="https://www.ipb.uni-bonn.de/people/xingguang-zhong/index.html"><strong>Xingguang Zhong</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/yue-pan/index.html"><strong>Yue Pan</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/cyrill-stachniss/"><strong>Cyrill Stachniss</strong></a>
    .
    <a href="https://www.ipb.uni-bonn.de/people/jens-behley/"><strong>Jens Behley</strong></a>
  </p>
  <p align="center"><a href="https://www.ipb.uni-bonn.de"><strong>University of Bonn</strong></a>
  <h3 align="center"><a href="https://arxiv.org/pdf/2401.09101v1.pdf">Paper</a> | Video</a></h3>
  <div align="center"></div>
</p>


![teaser](media/overview.jpg)
| Demo |
| :-: |
![teaser](media/4dndf.gif) |

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#run">How to run it</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>


## Abstract

<details>
  <summary>[Details (click to expand)]</summary>
Building accurate maps is a key building block to enable reliable localization, planning, and navigation of autonomous vehicles. We propose a novel approach for building accurate maps of dynamic environments utilizing a sequence of LiDAR scans. To this end, we propose encoding the 4D scene into a novel spatio-temporal implicit neural map representation by fitting a time-dependent truncated signed distance function to each point. Using our representation, we extract the static map by filtering the dynamic parts. Our neural representation is based on sparse feature grids, a globally shared decoder, and time-dependent basis functions, which we jointly optimize in an unsupervised fashion. To learn this representation from a sequence of LiDAR scans, we design a simple yet efficient loss function to supervise the map optimization in a piecewise way. We evaluate our approach on various scenes containing moving objects in terms of the reconstruction quality of static maps and the segmentation of dynamic point clouds. The experimental results demonstrate that our method is capable of removing the dynamic part of the input point clouds while reconstructing accurate and complete 3D maps, outperforming several state-of-the-art methods.
</details>

## Installation

We tested our code on Ubuntu 22.04 with an NVIDIA RTX 5000.

### 1. Set up conda environment

```
conda create --name 4dndf python=3.8
conda activate 4dndf
```

### 2. Install PyTorch

```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
```

The commands depend on your CUDA version. You may check the instructions [here](https://pytorch.org/get-started/previous-versions/).

### 3. Install PyTorch3D

Follow the official instructions [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) install PyTorch3D with conda.

### 4. Install other dependencies

```
pip install open3d==0.17 scikit-image tqdm pykdtree plyfile
conda install -c conda-forge quaternion
```

## How to run it

### Clone the repository
```
git clone https://gitlab.ipb.uni-bonn.de/starry.zhong/4dndf
cd 4dndf
```

### Sanity test and Demo

For a sanity test, run the following script to download the test data  (20 frames from KITTI seq 00) :
```
sh ./script/download_test_data.bash
```

Then run:
```
python static_mapping.py config/test/test.yaml
```
After training, It will generate a static mesh in `output/test` and visualize the dynamics segmentation result. For the visualizer, press 'space' to start the playback of the sequence (yellow points correspond to the static parts, and red points correspond to the identified dynamics).

## Evaluation
As reported in the paper, we evaluate our method in surface reconstruction and dynamic object segmentation.

### Surface reconstruction

We use [Co-fusion](https://github.com/martinruenz/co-fusion)'s car4 dataset and the Newer College dataset to evaluate the quality of our reconstruction. 
The original Co-fusion dataset provides depth images rendered in Blender. We convert these depth images into point clouds and use 150 frames for our experiments. (We also provide the script to convert the Co-fusion's data into our format in `script/cofusion_data_converter.py`.) Run the following script to download the already converted data: 

```
sh ./script/download_cofusion.bash
```

And then run the following script, which runs our pipeline (i.e., `static_mapping.py`) and computes the metrics:
```
sh ./script/run_cofusion.bash
```

The reconstructed mesh will be stored in the `output/cofusion` folder.

For the [newer college dataset](https://ori-drs.github.io/newer-college-dataset/),  we selected part of the data (1300 frames) in the yard for our experiment.
Run the following script to download the pre-processed data:

```
sh ./script/download_newer_college.bash
```

And then 
```
sh ./script/run_newer_college.bash
```
to run the training code and evaluation script.

The reconstructed mesh will be stored in the `output/newer_college` folder. Note that the ground truth point has a different cover area with input scans, so we provide the reference mesh to crop the reconstruction result automatically.

We also provide the meshes reconstructed by the baseline methods. You can download them by running:

```
sh /script/download_baseline.bash
```

Then change the path of `est_ply ` in `eval/eval_cofusion.py` and `eval/eval_newercollege.py` and run them to check the numbers.

### Dynamic objects segementation

We use [KTH_DynamicMap_Benchmark](https://github.com/KTH-RPL/DynamicMap_Benchmark) to evaluate the result of our Dynamic objects segmentation.

Download the data from the official link [here](https://zenodo.org/records/8160051) and unzip it to our data folder as:

```
./4dNDF/
└── data/
    └── KTH_dynamic/
            ├── 00/
            │   ├── gt_cloud.pcd
            │   ├── pcd/
            |   |    ├── 004390.pcd
            |   |    ├── 004391.pcd
            |   |    └── ...
            ├── 05/ 
            ├── av2/
            ├── semindoor/
            └── translations/
```

The benchmark doesn't explicitly provide the pose files, so we extract poses from the data and store them in the `data/translations/` folder.

For evaluating, you need to clone (to somewhere you like) and compile the [KTH_DynamicMap_Benchmark](https://github.com/KTH-RPL/DynamicMap_Benchmark) 's repo. The following commands should work if you have ROS-full installed on your machine.

```
git clone --recurse-submodules https://github.com/KTH-RPL/DynamicMap_Benchmark.git
cd DynamicMap_Benchmark/script
mkdir build && cd build
camke ..
make
```

Or check the guidance from the benchmark [here](https://github.com/KTH-RPL/DynamicMap_Benchmark/tree/master/methods).

Then, copy the Python file from `4dNDF/eval/evaluate_single_kth.py` to `/path/to/DynamicMap_Benchmark/scripts/py/eval/`

Take sequence 00 as an example. Run:

```
python static_mapping.py config/kth/00.yaml
```

After training, the static point cloud can be found here: `data/kth/00/static_points.pcd`

To evaluate it, run ( change the `/your/path/to` and `/path/to` to the correct path):

```
cd /your/path/to/DynamicMap_Benchmark/scripts/build/
./export_eval_pcd  /path/to/4dNDF/data/KTH_dynamic/00 static_points.pcd 0.05
```
It will generate the eval point cloud. Finally, Run:

```
python /your/path/to/DynamicMap_Benchmark/scripts/py/eval/evaluate_single_kth.py /path/to/4dNDF/data/KTH_dynamic/00
```
to check the number. For other sequences, we need to change all the `00` to `05`,` av2`, or `semindoor`. You can organize the commands as a bash script to make it more convenient.



## Citation
If you use 4dNDF for your academic work, please cite:
```
@inproceedings{zhong2024cvpr,
  author = {Xingguang Zhong and Yue Pan and Cyrill Stachniss and Jens Behley},
  title = {{3D LiDAR Mapping in Dynamic Environments using a 4D Implicit Neural Representation}},
  booktitle = {{Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR)}},
  year = 2024,
}
```

## Contact
If you have any questions, Feel free to contact:

- Xingguang Zhong {[zhong@igg.uni-bonn.de]()}
- Yue Pan {[yue.pan@igg.uni-bonn.de]()}
