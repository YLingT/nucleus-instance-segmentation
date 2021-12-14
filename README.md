# nucleus-instance-segmentation
This is the code for nucleus instance segmentation using mask rcnn with detectron2.

## Enviroment setting and dependencies 
Use pip install or conda install, and check the version :
```
#Name                        Version
python                       3.7.11
torch                        1.8.2
torchvision                  0.9.2
pycococreatortools           0.2.0
detectron2                   0.6
opencv-python                4.5.4.60
```

## Dataset 
There are 24 images for training and 6 images for testing, each image has more than 300 nucleus.

## Code 
### 0. Download Project
```
git clone https://github.com/YLingT/nucleus-instance-segmentation  
cd nucleus-instance-segmentation
```
### 1.  Data preparing
Run coco_format.py to generate coco format dataset.
```
python coco_format.py
```
You can directly skip to step 2 if you gitclone this project.
The project structure are as follows:
```
nucleus-instance-segmentation
|── coco_dataset
  |── train
  |── test
  |── cell_instance_segmentation.json
|── coco_format.py
|── train.py
|── test.py
```
### 2.  Training
Training parameter setting:
```
iteration          10000
batch size         1
learning rate      1E-2
optimizer          SGD
lr scheduler       StepLR
```
Write the custom config file (function):
```
cfg = get_cfg()  
cfg.DATALOADER.NUM_WORKERS = 2
  
# get configuration from model_zoo
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]

cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
cfg.SOLVER.WARMUP_FACTOR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.WARMUP_METHOD = 'linear'
cfg.SOLVER.BASE_LR = 0.01
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.gamma = 0.1
cfg.SOLVER.IMS_PER_BATCH = 1

# test
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  
cfg.TEST.DETECTIONS_PER_IMAGE = 500
# dataset
cfg.DATASETS.TRAIN = ('train')
cfg.OUTPUT_DIR = "coco_dataset/train"
```
Run code:
```
python train.py
```
The trained model will save in coco_dataset/train as model_final.pth.
### 3.  Testing
Download trained weight: [best.pt](), put it coco_dataset/train folder.  
Test and generate answer.json:
```
python test.py
```
The answer.json will save in rootpath.  
Architecture in answer.json:
```
[
    {
        "image_id": 1,
        "score": 1.0,
        "category_id": 1,
        "bbox": [
            975.84765625,
            304.2831726074219,
            17.07867431640625,
            16.744964599609375
        ],
        "segmentation": {
            "size": [
                1000,
                1000
            ],
            "counts": "P]jm05Qo04M1O2N2N100O10O100O1O1O2M3N^`7"
        }
    },...
```
### 4.  Result analysis

