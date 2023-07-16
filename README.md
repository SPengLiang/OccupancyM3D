# OccupancyM3D

<p align="center"> <img src='img/arch.png' align="center" height="300px"> </p>

### Note that this is an initial version, this repository needs to be further cleaned and refactored.

paper link: [[2305.15694] Learning Occupancy for Monocular 3D Object Detection](https://arxiv.org/abs/2305.15694)

## Installation

We employ the design and framework of  OpenPCDet, and follows the base setup in CaDDN, thanks for their great work! 

Therefore, please follow the installation steps in [OpenPCDet](./OpenPCDet/README.md).

## Getting Started

First, please follow the KITTI data file generation in [CaDDN](https://github.com/TRAILab/CaDDN/blob/master/docs/GETTING_STARTED.md)

We do not use the heavy deeplibV3 backbone. Instead, we use the pre-trained dla34 backbone from [DD3D](https://github.com/TRI-ML/dd3d). Note that the authors slightly modify the model. We re-map the key and provide it at [Google Drive](https://drive.google.com/file/d/1VRUFk0Bwwz60cDrgqbIHXaWubjFCPWKk/view?usp=sharing).


Then, train the model:

```shell
cd OccupancyM3D/OpenPCDet/tools

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --rdzv_endpoint=localhost:6400 --nproc_per_node=4  train.py --launcher pytorch  --cfg_file ./cfgs/kitti_models/OccupancyM3D.yaml \
  --sync_bn --workers 4 \
  --num_epochs_to_eval 20  \
  --extra_tag OccupancyM3D
```

Eval the model using the pre-trained model:
```shell
CUDA_VISIBLE_DEVICES=0 python test.py --cfg_file ./cfgs/kitti_models/OccupancyM3D.yaml \
  --extra_tag val \
  --batch_size 2 --workers 4 \
  --ckpt $pre-trained-model-path
```


## Pretrained Model

To ease the usage, we provide the pre-trained model at: [Google Drive](https://drive.google.com/file/d/1PBrpNVypZMNY3l2fPOs4LoVwQwep0u45/view?usp=sharing)

Here we give the comparison.

<table align="center">
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Car@BEV IoU=0.7</td>    
        <td colspan="3",div align="center">Car@3D IoU=0.7</td>  
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod</td> 
        <td div align="center">Hard</td> 
        <td div align="center">Easy</td> 
        <td div align="center">Mod</td> 
        <td div align="center">Hard</td>  
    </tr>
    <tr>
        <td div align="center">original paper</td>
        <td div align="center">35.72</td> 
        <td div align="center">26.60</td> 
        <td div align="center">23.68</td> 
        <td div align="center">26.87</td> 
        <td div align="center">19.96</td> 
        <td div align="center">17.15</td> 
    </tr>    
    <tr>
        <td div align="center">this repo</td>
        <td div align="center">36.26</td> 
        <td div align="center">26.25</td> 
        <td div align="center">23.22</td> 
        <td div align="center">28.64</td> 
        <td div align="center">19.84</td> 
        <td div align="center">17.77</td> 
    </tr>
</table>

## Visualization Results

Some good cases and bad cases (marked using arrows)

KITTI results

![](img/q1.png)

Waymo results

![](img/q2.png)
