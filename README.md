# Uncertainty-aware Mean Teacher for Source-free Unsupervised Domain Adaptive Object Detection

The official pyTorch implementation of the method of Uncertainty-aware Mean Teacher for Source-free Unsupervised Domain Adaptive Object Detection

![image](/imgs/flow_final_out.drawio.png)


## Installation

Please refer to [INSTALL.md](https://github.com/deeptibhegde/UncertaintyAwareMeanTeacher/blob/main/docs/INSTALL.md) for setup.


## Dataset preperation

1. Download the relevant datasets: [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) , [Waymo](https://waymo.com/intl/en_us/dataset-download-terms/) , [nuScenes](https://www.nuscenes.org/download)

2. Generate simulated adverse weather data for KITTI using [LISA](https://github.com/velatkilic/LISA)

3. Organize each folder inside [data](/data/) like the following


```
UncertaintyAwareMeanTeacher_SFUDA

├── data (main data folder)
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
|
|
│   ├── kitti-rain
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
|
|
│   ├── kitti-snow
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
|
|
│   ├── kitti-fog
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
|
|
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
|
|
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── pcdet_gt_database_train_sampled_xx/
│   │   │── pcdet_waymo_dbinfos_train_sampled_xx.pkl  
|
|

```
## Pre-trained models


We implement the proposed method for the object detector [SECOND-iou](/secondiou/) for several domain shift scenarios. You can find the folder of pretrained models [here](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/dhegde1_jh_edu/EpsIlDbB43VOnznWmM_K3BgB67CjI3ZNuG36FHCjkK6z2w?e=IB1dKj). Find specific model downloads and their corresponding config files below.


| SECOND-iou |
-------------------------------------------------
| Domain shift | Model file  | Configuration file |
| ----------- | ----------- | -------------------|
| Waymo  -> KITTI| [download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/dhegde1_jh_edu/EX91o3reI2tCig1y6nmC_x0Bu50oA7_UPfh8yiNq2O-6mw?e=nGhgOJ) | [link](SECOND-iou/tools/cfgs/kitti_models/secondiou_oracle_ros.yaml) |
| Waymo  -> KITTI-rain| [download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/dhegde1_jh_edu/EdbV2n4k16xPkjfWF06Icy0B2_fxx19mVENwydRKhxSemQ?e=CtjaC0) | [link](SECOND-iou/tools/cfgs/kitti_models/secondiou_oracle_ros.yaml) |
|  nuScenes -> KITTI| [download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/dhegde1_jh_edu/EbCNUDkBtKFDhtCh8Fz3yK4BkppnhspfF-UJXusvYGD4fQ?e=mXdMVa) | [link](SECOND-iou/tools/cfgs/kitti_models/secondiou_oracle_ros.yaml) |
|  nuScenes -> KITTI-rain| [download]() | [link](SECOND-iou/tools/cfgs/kitti_models/secondiou_oracle_ros.yaml) |



Go to [secondiou](secondiou) for environment setup and implementation details.



