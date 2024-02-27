# CenterRadarNet: Joint 3D Object Detection and Tracking Framework using 4D FMCW Radar



> *arXiv ([arXiv 2311.01423](https://arxiv.org/abs/2311.01423))*  



    @misc{cheng2023centerradarnet,
          title={CenterRadarNet: Joint 3D Object Detection and Tracking Framework using 4D FMCW Radar}, 
          author={Jen-Hao Cheng and Sheng-Yao Kuan and Hugo Latapie and Gaowen Liu and Jenq-Neng Hwang},
          year={2023},
          eprint={2311.01423},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }


## NEWS

[2024-03-xx] 


## Abstract
Robust perception is a vital component for ensuring safe autonomous and assisted driving. Automotive radar (77 to 81 GHz), which offers weather-resilient sensing, provides a complementary capability to the vision- or LiDAR-based autonomous driving systems. Raw radio-frequency (RF) radar tensors contain rich spatiotemporal semantics besides 3D location information. The majority of previous methods take in 3D (Doppler-range-azimuth) RF radar tensors, allowing prediction of an object's location, heading angle, and size in bird's-eye-view (BEV). However, they lack the ability to at the same time infer objects' size, orientation, and identity in the 3D space. To overcome this limitation, we propose an efficient joint architecture called CenterRadarNet, designed to facilitate high-resolution representation learning from 4D (Doppler-range-azimuth-elevation) radar data for 3D object detection and re-identification (re-ID) tasks. As a single-stage 3D object detector, CenterRadarNet directly infers the BEV object distribution confidence maps, corresponding 3D bounding box attributes, and appearance embedding for each pixel. Moreover, we build an online tracker utilizing the learned appearance embedding for re-ID. CenterRadarNet achieves the state-of-the-art result on the K-Radar 3D object detection benchmark. In addition, we present the first 3D object-tracking result using radar on the K-Radar dataset V2. In diverse driving scenarios, CenterRadarNet shows consistent, robust performance, emphasizing its wide applicability.




## Main results

#### 3D detection on K-Radar test set

#### 3D detection on CRUW3D test set

#### 3D detection on Waymo test set

|         |  #Frame | Veh_L2 | Ped_L2 | Cyc_L2  | MAPH   |  FPS  |
|---------|---------|--------|--------|---------|--------|-------|
|VoxelNet | 1       |  71.9     |  67.0      |  68.2       |   69.0     |   13    | 
|VoxelNet | 2       |  73.0     |  71.5      |  71.3       |   71.9     |  11     |




## Installation
Please refer to [CenterPoint](https://github.com/tianweiy/CenterPoint?tab=readme-ov-file#installation) installation instruction.



### Benchmark Evaluation and Training 

Please refer to [GETTING_START](docs/GETTING_START.md) to prepare the data. Then follow the instruction there to reproduce our detection and tracking results. All detection configurations are included in [configs](configs).



## License

MIT license

CenterRadarNet is developed based on [CenterPoint](https://github.com/tianweiy/CenterPoint)'s code base.



## Acknowlegement
We greatly thank the following open-sourced projects which provides the code base for CenterRadarNet.

* [CenterPoint](https://github.com/tianweiy/CenterPoint) 
* [K-Radar dataset dev kit](https://github.com/kaist-avelab/K-Radar) 


# Dev Notes
- need to remove old test_kitti folder for reevaluation on the KRadar dataset