from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
import os
import numpy as np
import json
import cv2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import glob
import pycocotools.mask as mask_util


path_json = "coco_dataset/cell_instance_segmentation.json"
path_img = "coco_dataset/train"
for d in ["train"]:
    register_coco_instances(d, {}, path_json, path_img)
    MetadataCatalog.get(d).set(thing_classes=["cell"])
metadata = MetadataCatalog.get("train")


def custom_config():
    cfg = get_cfg()

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.OUTPUT_DIR = "coco_dataset/train"

    # get configuration from model_zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    # input
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    cfg.INPUT.RANDOM_FLIP = "vertical"

    # Model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0, 4.0, 8.0]]
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14

    cfg.DETECTION_MASK_THRESHOLD = 0.35
    cfg.DETECTIONS_PER_IM = 500

    # Solver
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 2000
    cfg.SOLVER.WARMUP_METHOD = 'linear'

    # Optimizer
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.STEPS = (5000, 8000, 9000)
    cfg.SOLVER.gamma = 0.1
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.WEIGHT_DECAY = 0.0001

    # Test
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    cfg.TEST.DETECTIONS_PER_IMAGE = 500

    # DATASETS
    cfg.DATASETS.TEST = ()
    cfg.DATASETS.TRAIN = ('train')

    return cfg


cfg = custom_config()
predictor = DefaultPredictor(cfg)


def GetMask(img):
    outputs = predictor(img)
    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return v.get_image()[:, :, ::-1]


test_path = sorted(glob.glob("coco_dataset/test/*png"))
for i in range(len(test_path)):
    print(test_path[i])
    im = cv2.imread(test_path[i])
    cv2.imshow("img", GetMask(im))
    cv2.waitKey(0)

data = []
for name in test_path:
        print(name)
        image = cv2.imread(name)
        outputs = predictor(image)
        image_id = int(os.path.basename(name).replace(".png", ""))
        print(image_id)

        masks = outputs["instances"].to('cpu').pred_masks.cpu().numpy()

        num = len(masks)
        scores = outputs["instances"].to('cpu').scores.cpu().numpy()
        boxes = outputs["instances"].to('cpu').pred_boxes.tensor.cpu().numpy()

        for i in range(num):
            a = {}

            a["image_id"] = image_id
            a["score"] = float(scores[i])
            a["category_id"] = 1

            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2] - box[0]
            height = box[3] - box[1]

            a["bbox"] = (tuple((float(left), float(top), float(width), float(height))))

            segmentation = {}
            segmentation["size"] = [1000, 1000]
            mask = masks[i]

            segmentation["counts"] = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]["counts"].decode('utf-8')
            a["segmentation"] = segmentation
            data.append(a)

ret = json.dumps(data, indent=4)
print(len(data))
with open("answer.json", 'w') as fp:
    fp.write(ret)
