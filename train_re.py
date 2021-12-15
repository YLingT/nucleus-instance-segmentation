import torch
import os
import json
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import DatasetEvaluators
from detectron2.solver.build import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping


def load_data(t="train"):
    if t == "train":
        with open("coco_dataset/cell_instance_segmentation.json", 'r', encoding='utf-8') as file:
            train = json.load(file)
        return train


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params = get_default_optimizer_params(model,
                                              base_lr=cfg.SOLVER.BASE_LR,
                                              weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
                                              bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
                                              weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
                                              )
        return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(params,
                                                                  lr=cfg.SOLVER.BASE_LR,
                                                                  betas=cfg.SOLVER.BETAS,
                                                                  eps=cfg.SOLVER.EPS,
                                                                  weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                                                                  )


def custom_config():
    cfg = get_cfg()

    cfg.DATALOADER.NUM_WORKERS = 2

    # get configuration from model_zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

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

    # DATASETS
    cfg.OUTPUT_DIR = "coco_dataset/train"

    return cfg


if __name__ == '__main__':
    dataset_name = 'train'
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)
    path_json = r"coco_dataset/cell_instance_segmentation.json"
    path_img = r"coco_dataset/train"
    for d in ["train"]:
        register_coco_instances(d, {}, path_json, path_img)
        MetadataCatalog.get(d).set(thing_classes=["cell"])
    metadata = MetadataCatalog.get("train")

    cfg = custom_config()

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()
