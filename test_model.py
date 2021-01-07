#!/bin/python3

import os

import numpy as np
import cv2
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog,Metadata
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import torch

PREFIX_DIR = '/home/jcp353/bgnn_data/'
#IMAGES_DIR = 'INHS_segmented_padded_fish/full_imgs/'
IMAGES_DIR = 'full_imgs_large/'
#IMAGES_DIR = 'white_background_fish/'
IMS = PREFIX_DIR + IMAGES_DIR

cfg = get_cfg()
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = ('./model_final.pth')
#model = build_model(cfg)
#DetectionCheckpointer(model).load('./model_final.pth')
#model.load_state_dict(torch.load('./model_final.pth', map_location='cpu'))
#model='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
#cfg.merge_from_file(model_zoo.get_config_file(
#    'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
#cfg.merge_from_file(model_zoo.get_config_file(model))
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
#        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

cfg.DATASETS.TRAIN = ('fish_train',)
cfg.DATASETS.TEST = ('fish_train',)

metadata=Metadata(evaluator_type='coco', image_root=IMS, json_file=PREFIX_DIR+'fish_train.json', name='fish_train')
#thing_classes=['fish'], thing_dataset_id_to_contiguous_id={1: 0})

predictor = DefaultPredictor(cfg)
outputs = []
segments = os.listdir(IMS)
names = [i.split('.')[0] for i in segments]
#names = ['INHS_FISH_00190', 'INHS_FISH_000314', 'INHS_FISH_00357']
random.shuffle(names)
j=0
while j < 20:
    curr_img = IMS + names[j] + '.jpg'
    im = cv2.imread(curr_img)
    print(f'{j}: {curr_img}')
    outputs.append(predictor(im))
    # Scaling the image 1.5 times, for big images consider a value below 0
    v = Visualizer(im, metadata, scale=1.0)
    v = Visualizer(im, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    v = v.draw_instance_predictions(outputs[j]['instances'].to('cpu'))
    cv2.imwrite(f'./test_images_output/{names[j]}.jpg', v.get_image())
    j+=1
