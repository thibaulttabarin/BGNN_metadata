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
import pprint
import train_detectron as td

pp = pprint.PrettyPrinter()

PREFIX_DIR = '/home/HDD/bgnn_data/'
IMAGES_DIR = 'blue_gill/'
IMS = PREFIX_DIR + IMAGES_DIR

cfg = get_cfg()
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model
model='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

cfg.DATASETS.TRAIN = ('fish_train',)
cfg.DATASETS.TEST = ('fish_train',)

metadata=Metadata(evaluator_type='coco', image_root=IMS, json_file=PREFIX_DIR+'fish_train.json', name='fish_train',
thing_classes=['F', 'F' ,'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'fish'])
#thing_classes=['fish'], thing_dataset_id_to_contiguous_id={14: 1})

predictor = DefaultPredictor(cfg)
outputs = []
segments = os.listdir(IMS)
names = [i.split('.')[0] for i in segments]
#names = ['INHS_FISH_100172']
#names = ['INHS_FISH_00190', 'INHS_FISH_000314', 'INHS_FISH_00357']
#print(names)
random.shuffle(names)
i=0
j=0
while j < 10:
    curr_img = names[i] + '.jpg'
    #curr_img = 'OSUM0000001.JPG'
    #print(curr_img)
    im = cv2.imread(IMS + curr_img)
    #im = cv2.resize(im, (800, 600))
    outputs.append(predictor(im))
    #if outputs[-1]['instances'].num_instances > 0:
    #if True:
    bbox = [float(x) for x in outputs[-1]['instances'].get('pred_boxes').tensor[0]]
    #pp.pprint(outputs[-1]['instances'])
    #pp.pprint(bbox)
    x_add = (bbox[2] - bbox[0]) * 0.02
    y_add = (bbox[3] - bbox[1]) * 0.02
    bbox[0] -= x_add
    bbox[2] += x_add
    bbox[1] -= y_add
    bbox[3] += y_add
    #print(x_add)
    #print(y_add)
    #exit(0)
    if len(outputs[-1]['instances']):
        td.gen_temp3(bbox, curr_img)
        #td.gen_temp3(bbox, 'OSUM0000001.JPG')
        # Scaling the image 1.5 times, for big images consider a value below 0
        v = Visualizer(im, None, scale=1.0)
        v = Visualizer(im, metadata, scale=1.0)
        try:
            v = v.draw_instance_predictions(outputs[i]['instances'].to('cpu'))
            #cv2.imwrite(f'/home/joel/School/Research/bgnn/drexel_metadata/test_images_output/{curr_img}', v.get_image())
            cv2.imwrite(f'image_output/{names[i]}_detectron.jpg', v.get_image())
            j+=1
            print(f"{list(outputs[i]['instances'].get('pred_classes')).count(14)} fish found")
            print(f'Output image #{j}')
        except Exception:
            print('0 fish found (or were mislabeled)')
    i+=1
