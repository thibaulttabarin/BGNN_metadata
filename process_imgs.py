#!/bin/python3

import os
from multiprocessing import Pool
import pandas as pd
import numpy as np
import nrrd
from PIL import Image
from functools import partial
from random import shuffle

import torch

import pycocotools
import detectron2.structures as structures
import detectron2.data.datasets.coco as coco
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog,\
        build_detection_train_loader,\
        build_detection_test_loader
from detectron2.engine.defaults import DefaultTrainer,\
        default_argument_parser
from detectron2.engine import launch
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from skimage import filters
from skimage.morphology import flood_fill
from copy import copy

PREFIX_DIR = '/home/jcp353/bgnn_data/'
IMAGES_DIR = 'full_imgs_large/'

def wrapper2(df, i):
    name = df['Name'][i]
    l = df['x1'][i]
    r = df['x2'][i]
    t = df['y1'][i]
    b = df['y2'][i]
    print(f'{i}: {name}')

    try:
        im = Image.open(f'{PREFIX_DIR}{IMAGES_DIR}{name}').convert('L')
        arr2 = np.array(im)
        shape = arr2.shape
        bbox = (l,t,r,b)
        arr0 = np.array(im.crop(bbox))
        bb_size = arr0.size

        val = filters.threshold_otsu(arr0) * 1.4
        arr1 = np.where(arr0 < val, 1, 0).astype(np.uint8)
        indicies = list(zip(*np.where(arr1 == 1)))
        shuffle(indicies)
        for ind in indicies:
            temp = flood_fill(arr1, ind, 2)
            temp = np.where(temp == 2, 1, 0)
            percent = np.count_nonzero(temp) / bb_size
            #print(percent)
            if percent > 0.1:
                temp = flood_fill(temp, (0, 0), 2)
                arr1 = np.where(temp != 2, 255, 0).astype(np.uint8)
                break
        arr2[t:b,l:r] = arr1
        im2 = Image.fromarray(arr2, 'L')
        im2.save(f"/home/jcp353/out_imgs/{name}")
        return gen_dict(arr1, name, list(bbox))
    except FileNotFoundError:
        return None

def gen_coco_dataset2():
    df = pd.read_csv(f'{PREFIX_DIR}inhs_bboxes.csv', sep=' ')
    #name = 'INHS_FISH_000452.jpg'
    #i = (df['Name']=='INHS_FISH_000452.jpg').idxmax()
    for i in range(30):
        wrapper2(df, i)
    exit(0)
    #f = partial(wrapper2, df)
    #output = [f(0)]
    #with Pool() as p:
        #output = p.map(f, list(range(len(df))))
    #return [x for x in output if x is not None]

def gen_dict(mask, name, bbox):
    fish_dict = {}
    fish_dict['file_name'] = f'{PREFIX_DIR}{IMAGES_DIR}{name}'
    fish_dict['height'], fish_dict['width'] = mask.shape
    fish_dict['image_id'] = name.split('.')[0]
    annotate = {}
    annotate['bbox'] = bbox
    #print(annotate['bbox'])
    annotate['bbox_mode'] = structures.BoxMode.XYXY_ABS
    annotate['category_id'] = 0
    annotate['segmentation'] = \
            pycocotools.mask.encode(np.asfortranarray(mask))
    fish_dict['annotations'] = [annotate]
    return fish_dict

def gen_dataset_json():
    DatasetCatalog.register('fish', gen_coco_dataset2)
    MetadataCatalog.get('fish').set(thing_classes=['fish'])
    out_file = PREFIX_DIR + 'fish_train2.json'
    print(f'Saving to file {out_file}')
    coco.convert_to_coco_json('fish', out_file)

if __name__ == '__main__':
    #gen_dataset_json()
    gen_coco_dataset2()
