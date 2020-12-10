#!/bin/python3

import os
from multiprocessing import Pool
import pandas as pd
import numpy as np
import nrrd
from PIL import Image
from multiprocessing import Pool

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

PREFIX_DIR = '/home/jcp353/bgnn_data/'
IMAGES_DIR = 'full_imgs_large/'
LM_DIR = 'labelmaps/validation/'
SEGS = PREFIX_DIR + LM_DIR

def gen_mask(data):
    out = np.zeros((data.shape[1], data.shape[2])).astype(np.uint8)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if data[0][i][j] != 0:
                out[i][j] = 1
    return out

def gen_mask_wrapper(name, segs):
    curr_file = segs + name + '.nrrd'
    try:
        data, _ = nrrd.read(curr_file, index_order='C')
        if data.shape == (1, 1, 1):
            print(f'Error: bad nrrd file {curr_file}')
        else:
            print(f'{name}: {data.shape}')
            return gen_dict(gen_mask(data), name)
    except FileNotFoundError:
        print(f'Error: could not find and/or output {curr_file}')
    return None

def lambda_wrapper(name):
    return gen_mask_wrapper(name, SEGS)

def gen_coco_dataset2():
    df = pd.read_csv(f'{PREFIX_DIR}inhs_bboxes.csv', sep=' ')
    output = []
    for i in range(len(df)):
        print(i)
        name = df['Name'][i]
        l = df['x1'][i]
        r = df['x2'][i]
        t = df['y1'][i]
        b = df['y2'][i]

        im = Image.open(f'{PREFIX_DIR}{IMAGES_DIR}{name}').convert('L')
        shape = np.array(im).shape
        bbox = (l,t,r,b)
        arr0 = np.array(im.crop(bbox))

        val = filters.threshold_otsu(arr0)
        arr1 = np.where(arr0 < val, 1, 0).astype(np.uint8)
        arr2 = np.full(shape, 0).astype(np.uint8)
        arr2[t:b,l:r] = arr1
        #im2 = Image.fromarray(arr2, 'L')
        #im2.save("/home/joel/test_out.jpg")
        output.append(gen_dict(arr2, name, list(bbox)))
    return output

def gen_coco_dataset():
    segments = os.listdir(PREFIX_DIR + LM_DIR)
    names = [i.split('.')[0] for i in segments]
    with Pool() as p:
        return p.map(lambda_wrapper, names)

def gen_dict(mask, name, bbox):
    fish_dict = {}
    fish_dict['file_name'] = f'{PREFIX_DIR}{IMAGES_DIR}{name}.jpg'
    fish_dict['height'], fish_dict['width'] = mask.shape
    fish_dict['image_id'] = name
    annotate = {}
    #annotate['bbox'] = bbox(mask)
    annotate['bbox'] = bbox
    annotate['bbox_mode'] = structures.BoxMode.XYXY_ABS
    annotate['category_id'] = 0
    annotate['segmentation'] = \
            pycocotools.mask.encode(np.asfortranarray(mask))
    fish_dict['annotations'] = [annotate]
    return fish_dict

#https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    #print(mask)
    #print(rows)
    #print(cols)
    #exit(0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return [float(x) for x in [rmin, cmin, rmax, cmax]]

class Trainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg,
           mapper=DatasetMapper(cfg, is_train=True, augmentations=[
               T.Resize((800, 600))
           ]))

    #@classmethod
    #def build_test_loader(cfg, dataset_name):
        #return build_detection_test_loader(cfg)

def gen_dataset_json():
    DatasetCatalog.register('fish', gen_coco_dataset2)
    MetadataCatalog.get('fish').set(thing_classes=['fish'])
    out_file = PREFIX_DIR + 'fish_train.json'
    print(f'Saving to file {out_file}')
    coco.convert_to_coco_json('fish', out_file)

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

def white_out_background():
    #data, _ = nrrd.read(PREFIX_DIR + LM_DIR + 'INHS_FISH_51445.nrrd',
            #index_order='C')
    #mask = gen_mask(data)
    #fdict = gen_dict(mask, 'INHS_FISH_51445')
    #print(fdict)
    #print_image_outline('INHS_FISH_51445')
    #process_nrrds()
    #name = 'INHS_FISH_2401.nrrd'
    #curr_file = PREFIX_DIR + LM_DIR + name
    #print(gen_mask(data).shape)
    segments = os.listdir(PREFIX_DIR + LM_DIR)
    names = [i.split('.')[0] for i in segments]
    with Pool() as p:
        images = p.map(f, names)
    for i in range(len(images)):
        curr_img = PREFIX_DIR + IMAGES_DIR + names[i] + '.jpg'
        im2 = Image.fromarray(images[i])
        im2.save(curr_img)
        #if data.shape[2] != im.width or data.shape[1] != im.height:
        #    print(f'image: {im.width}, {im.height}')
        #    print(f'nrrd: {data.shape[2]}, {data.shape[1]}')
        #    print(name)

def check_size():
    #data, _ = nrrd.read(PREFIX_DIR + LM_DIR + 'INHS_FISH_51445.nrrd',
        #index_order='C')
    #mask = gen_mask(data)
    #fdict = gen_dict(mask, 'INHS_FISH_51445')
    #print(fdict)
    #print_image_outline('INHS_FISH_51445')
    #process_nrrds()
    #name = 'INHS_FISH_2401.nrrd'
    #curr_file = PREFIX_DIR + LM_DIR + name
    #print(gen_mask(data).shape)
    segments = os.listdir(PREFIX_DIR + LM_DIR)
    names = [i.split('.')[0] for i in segments]
    for name in names:
        curr_nrrd = PREFIX_DIR + LM_DIR + name + '.nrrd'
        curr_img = PREFIX_DIR + IMAGES_DIR + name + '.jpg'
        data, _ = nrrd.read(curr_nrrd, index_order='C')
        im = Image.open(curr_img)
        if data.shape[2] != im.width or data.shape[1] != im.height:
            print(f'image: {im.width}, {im.height}')
            print(f'nrrd: {data.shape[2]}, {data.shape[1]}')
            print(name)

def train():
    cfg = get_cfg()
    val_file = PREFIX_DIR + 'fish_val.json'
    train_file = PREFIX_DIR + 'fish_train.json'
    register_coco_instances('fish_val', {}, val_file, PREFIX_DIR + IMAGES_DIR)
    register_coco_instances('fish_train', {}, train_file,
            PREFIX_DIR + IMAGES_DIR)
    cfg.DATASETS.TRAIN = ('fish_train',)
    cfg.DATASETS.TEST = ('fish_val',)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 100
    cfg.SOLVER.MAX_ITER = 125000
    #cfg.DATALOADER.NUM_WORKERS = 2
    #cfg.INPUT.MIN_SIZE_TRAIN = (100,)
    trainer = Trainer(cfg)
    torch.cuda.empty_cache()
    trainer.resume_or_load()
    trainer.train()
    return trainer.train()

if __name__ == '__main__':
    #white_out_background()
    #launch(train, 2)
    gen_dataset_json()
    #check_size()
