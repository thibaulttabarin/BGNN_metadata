#!/bin/python3

import os

import numpy as np
import cv2
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import torch

PREFIX_DIR = '/home/HDD/bgnn_data/'
IMAGES_DIR = 'INHS_segmented_padded_fish/'
IMS = PREFIX_DIR + IMAGES_DIR

cfg = get_cfg()
model = build_model(cfg)
DetectionCheckpointer(model).load('./model_final0.pth')
#model.load_state_dict(torch.load('./model_final.pth', map_location='cpu'))
#model='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
#cfg.merge_from_file(model_zoo.get_config_file(
#    'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
#cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
#        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

cfg.DATASETS.TRAIN = ('fish_train',)

predictor = DefaultPredictor(cfg)
outputs = []
segments = os.listdir(IMS)
names = [i.split('.')[0] for i in segments]
random.shuffle(names)
for i in range(20):
    curr_img = IMS + names[i] + '.jpg'
    im = cv2.imread(curr_img)
    outputs.append(predictor(im))
    # Scaling the image 1.5 times, for big images consider a value below 0
    v = Visualizer(im, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
    v = v.draw_instance_predictions(outputs[i]['instances'].to('cpu'))
    print("HERE")
    cv2.imwrite(f'./output/{names[i]}.jpg', v.get_image())
