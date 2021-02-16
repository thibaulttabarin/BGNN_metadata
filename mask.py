#!/bin/python3

import os
from multiprocessing import Pool
import pandas as pd
import numpy as np
import nrrd
from PIL import Image
from functools import partial

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
from random import shuffle

def gen_temp3(bbox, name):
    #print(bbox)
    print(name)
    l = round(bbox[0])
    r = round(bbox[2])
    t = round(bbox[1])
    b = round(bbox[3])

    im = Image.open(f'{PREFIX_DIR}{IMAGES_DIR}{name}').convert('L')
    arr2 = np.array(im)
    shape = arr2.shape
    bbox = (l,t,r,b)
    arr0 = np.array(im.crop(bbox))
    bb_size = arr0.size

    val = filters.threshold_otsu(arr0) * 1.4
    #val = filters.threshold_otsu(arr0) * 0.75
    arr1 = np.where(arr0 < val, 1, 0).astype(np.uint8)
    #arr1 = np.where(arr0 > val, 1, 0).astype(np.uint8)
    indicies = list(zip(*np.where(arr1 == 1)))
    shuffle(indicies)
    count = 0
    for ind in indicies:
        count += 1
        if count > 10000:
            print(f'ERROR on flood fill: {name}')
            return None
        temp = flood_fill(arr1, ind, 2)
        temp = np.where(temp == 2, 1, 0)
        percent = np.count_nonzero(temp) / bb_size
        #print(percent)
        if percent > 0.1:
            temp = flood_fill(temp, (0, 0), 2)
            arr1 = np.where(temp != 2, 1, 0).astype(np.uint8)
            break
    arr3 = np.full(shape, 0).astype(np.uint8)
    arr3[t:b,l:r] = arr1
    #arr3 = arr1
    arr3 = np.where(arr3 == 1, 255, 0).astype(np.uint8)
    im2 = Image.fromarray(arr3, 'L')
    im2.save(f'image_output/THIS_{name.split(".")[0]}_pixel_mask.jpg')
    exit(0)

def gen_coco_dataset2():
    df = pd.read_csv(f'{PREFIX_DIR}boxes.csv', sep=',')
    #f = partial(wrapper2, df)
    f = partial(wrapper3, df)
    #output = [f(0)]
    #with Pool() as p:
        #output = p.map(f, list(range(len(df))))
        #output = p.map(f, list(range(10)))
    #return [x for x in output if x is not None]
    with Pool(1) as p:
        output = map(f, df['Name'].unique())
        #output = p.map(f, list(range(len(df))))
        #output = p.map(f, list(range(1000)))
    g = [x for x in output if x is not None]
    print(g[0])
    return g

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
        count = 0
        for ind in indicies:
            count += 1
            if count > 10000:
                print(f'ERROR on flood fill: {name}')
                return None
            temp = flood_fill(arr1, ind, 2)
            temp = np.where(temp == 2, 1, 0)
            percent = np.count_nonzero(temp) / bb_size
            #print(percent)
            if percent > 0.1:
                temp = flood_fill(temp, (0, 0), 2)
                arr1 = np.where(temp != 2, 1, 0).astype(np.uint8)
                break
        arr3 = np.full(shape, 0).astype(np.uint8)
        arr3[t:b,l:r] = arr1
        #im2 = Image.fromarray(arr2, 'L')
        #im2.save(f"/home/jcp353/out_imgs/{name}")
        return gen_dict(arr3, name, list(bbox))
    except FileNotFoundError:
        return None

def gen_coco_dataset():
    segments = os.listdir(PREFIX_DIR + LM_DIR)
    names = [i.split('.')[0] for i in segments]
    with Pool() as p:
        return p.map(lambda_wrapper, names)

#https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    #print(mask)
    #print(rows)
    #print(cols)
    #exit(0)
    try:
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
    except:
        return None

    return [float(x) for x in [cmin, rmin, cmax, rmax]]

def f(name):
    #curr_nrrd = PREFIX_DIR + LM_DIR + name + '.nrrd'
    curr_img = PREFIX_DIR + IMAGES_DIR + name + '.jpg'
    #data, _ = nrrd.read(curr_nrrd, index_order='C')
    x = np.array(Image.open(curr_img))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not x[i][j].any():
                x[i][j] = np.array([255,255,255])
    return x

