# CenterRadarNet: Joint 3D Object Detection and Tracking Framework using 4D FMCW Radar






## What is CenterRadarNet?
Demo: https://youtu.be/KxCib_1JkoI?si=DWDVP85qgOPUJZDZ

Robust perception is a vital component for ensuring safe autonomous and assisted driving. Automotive radar (77 to 81 GHz), which offers weather-resilient sensing, provides a complementary capability to the vision- or LiDAR-based autonomous driving systems. Raw radio-frequency (RF) radar tensors contain rich spatiotemporal semantics besides 3D location information. The majority of previous methods take in 3D (Doppler-range-azimuth) RF radar tensors, allowing prediction of an object's location, heading angle, and size in bird's-eye-view (BEV). However, they lack the ability to at the same time infer objects' size, orientation, and identity in the 3D space. To overcome this limitation, we propose an efficient joint architecture called CenterRadarNet, designed to facilitate high-resolution representation learning from 4D (Doppler-range-azimuth-elevation) radar data for 3D object detection and re-identification (re-ID) tasks. As a single-stage 3D object detector, CenterRadarNet directly infers the BEV object distribution confidence maps, corresponding 3D bounding box attributes, and appearance embedding for each pixel. Moreover, we build an online tracker utilizing the learned appearance embedding for re-ID. CenterRadarNet achieves the state-of-the-art result on the K-Radar 3D object detection benchmark. In addition, we present the first 3D object-tracking result using radar on the K-Radar dataset V2. In diverse driving scenarios, CenterRadarNet shows consistent, robust performance, emphasizing its wide applicability.







## Installation

```bash
git clone https://github.com/Andy-Cheng/CenterRadarNet.git
cd CenterRadarNet
git checkout kradar
```

## Basic Installation
```bash
conda create -n cenrad python=3.9
conda activate cenrad
pip install -r requirements.txt
pip install -r requirements-torch.txt

# add CenterRadarNet to PYTHONPATH by adding the following line to ~/.bashrc (change the path accordingly)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_CenterRadarNet"
```

## Cuda Extension

```bash
# set the cuda path(change the path to your own cuda location) 
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8

# Rotated NMS 
cd det3d/ops/iou3d_nms
python setup.py build_ext --inplace
cd ../../..
```

## APEX (Optional)
```bash
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 5633f6  # recent commit doesn't build in our system 
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```

## Dataset Preparation
Please refer to the [K-Radar dataset dev kit](https://github.com/kaist-avelab/K-Radar) for the dataset preparation.



## Training

Please use the following command as an example to train CenterRadarNet on K-Radar dataset.
```bash
bash scripts/train_example.sh
```



## Inference

```bash
python tools/dist_test.py configs/kradar/exps/hr3d_allseq.py --work_dir work_dirs/kradar_hr3d --checkpoint work_dirs/kradar_hr3d/epoch_22.pth --testset
```

## Dev Notes

- The multi-modal version is in the `corss_modal` branch, and is currently under development.

## Citation

Please cite the following paper if you find our codes or idea useful!!!

> *arXiv ([arXiv 2311.01423](https://arxiv.org/abs/2311.01423))*  



    @INPROCEEDINGS{10648077,
    author={Cheng, Jen-Hao and Kuan, Sheng-Yao and Liu, Hou-I and Latapie, Hugo and Liu, Gaowen and Hwang, Jenq-Neng},
    booktitle={2024 IEEE International Conference on Image Processing (ICIP)}, 
    title={CenterRadarNet: Joint 3D Object Detection and Tracking Framework Using 4D FMCW Radar}, 
    year={2024},
    volume={},
    number={},
    pages={998-1004},
    keywords={Radio frequency;Three-dimensional displays;Tensors;Radar detection;Radar;Object detection;Benchmark testing;Automotive Radar;3D Object Detection;Multi-Object Tracking;Autonomous Driving},
    doi={10.1109/ICIP51287.2024.10648077}}



## License
MIT license

## Acknowlegement
We greatly thank the following open-sourced projects which provides the code base for CenterRadarNet.

* [CenterPoint](https://github.com/tianweiy/CenterPoint) 
* [K-Radar dataset dev kit](https://github.com/kaist-avelab/K-Radar) 


