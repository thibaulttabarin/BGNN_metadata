#!/bin/python3

import os

import numpy as np
import nrrd
from PIL import Image

import torch

import pycocotools
import detectron2.structures as structures
import detectron2.data.datasets.coco as coco
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine.defaults import DefaultTrainer
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.config import get_cfg

PREFIX_DIR = '/home/HDD/bgnn_data/'
IMAGES_DIR = 'INHS_segmented_padded_fish/'
SEGS_DIR = 'segmentations_wt/'
LM_DIR = 'labelmaps/validation/'

def process_nrrds():
    segments = os.listdir(PREFIX_DIR + LM_DIR)
    names = [i.split('.')[0] for i in segments]
    segs = PREFIX_DIR + LM_DIR
    errored = set()
    for name in names:
        curr_file = segs + name + '.nrrd'
        try:
            data, _ = nrrd.read(curr_file, index_order='C')
            if data.shape == (1, 1, 1):
                errored.add(name)
            else:
                if data.shape[0] < 4:
                    print(f'{name}: {data.shape}')
                    print_nrrd_map(data, name)
        except FileNotFoundError:
            print(f'Error: could not find {curr_file}')
            errored.add(name)
    print(len(set(names).difference(errored)))

def print_image_mask(name):
    try:
        orig = Image.open(PREFIX_DIR + IMAGES_DIR + name + '.jpg')
    except FileNotFoundError:
        orig = Image.open(PREFIX_DIR + IMAGES_DIR + name + '.JPG')
    orig = orig.convert('L')
    orig = np.array(orig).astype(np.int32)
    out = np.full(orig.shape, 255).astype(np.int32)
    for i in range(orig.shape[0]):
        for j in range(orig.shape[1]):
            d = orig[i][j]
            if d < 240 and d > 45:
                out[i][j] = 0
    img = Image.fromarray(out.astype(np.uint8))
    img.save(PREFIX_DIR + 'auto_masks/' + name + '.png')

def print_image_outline(name):
    try:
        orig = Image.open(PREFIX_DIR + IMAGES_DIR + name + '.jpg')
    except FileNotFoundError:
        orig = Image.open(PREFIX_DIR + IMAGES_DIR + name + '.JPG')
    orig = orig.convert('L')
    orig = np.array(orig).astype(np.int32)
    for size in range(5, 200, 5):
        print(size)
        out = np.full(orig.shape, 255).astype(np.int32)
        for i in range(orig.shape[0]):
            for j in range(orig.shape[1]):
                d = orig[i][j]
                for k in range(-1, 2):
                    for w in range(-1, 2):
                        try:
                            if d-orig[i+k][j+w] > size:
                                out[i][j] = 0
                        except IndexError:
                            # Just means we are at an edge
                            pass
        img = Image.fromarray(out.astype(np.uint8))
        img.save(PREFIX_DIR + 'outlines/' + name + f'_{size}.png')

def gen_mask(data):
    out = np.zeros((data.shape[1], data.shape[2])).astype(np.uint8)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if data[0][i][j] != 0:
                out[i][j] = 1
    return out

def print_nrrd_map(data, name):
    out = np.full((data.shape[1], data.shape[2]), 255).astype(np.uint8)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if data[0][i][j] != 0:
                out[i][j] = 0
    img = Image.fromarray(out)
    img.save(PREFIX_DIR + 'maps_lm/' + name + '.png')

def gen_coco_dataset():
    segments = os.listdir(PREFIX_DIR + LM_DIR)
    names = [i.split('.')[0] for i in segments]
    segs = PREFIX_DIR + LM_DIR
    errored = set()
    count = 1
    total = len(names)
    dataset = []

    for name in names:
        curr_file = segs + name + '.nrrd'
        try:
            data, _ = nrrd.read(curr_file, index_order='C')
            if data.shape == (1, 1, 1):
                errored.add(name)
            else:
                print(f'{count} / {total} | {name}: {data.shape}')
                count += 1
                dataset.append(gen_dict(gen_mask(data), name))
        except FileNotFoundError:
            print(f'Error: could not find {curr_file}')
            errored.add(name)
    print(f'Errored: {errored}')
    return dataset

def gen_dict(mask, name):
    fish_dict = {}
    fish_dict['file_name'] = f'{PREFIX_DIR}{IMAGES_DIR}{name}.jpg'
    fish_dict['height'], fish_dict['width'] = mask.shape
    fish_dict['image_id'] = name
    annotate = {}
    annotate['bbox'] = bbox(mask)
    annotate['bbox_mode'] = structures.BoxMode.XYXY_ABS
    annotate['category_id'] = 0
    annotate['segmentation'] = \
            pycocotools.mask.encode(np.asarray(mask, order="F"))
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

def overlay_mask_on_image(data, name):
    try:
        orig = Image.open(PREFIX_DIR + IMAGES_DIR + name + '.jpg')
    except FileNotFoundError:
        orig = Image.open(PREFIX_DIR + IMAGES_DIR + name + '.JPG')
    out = np.array(orig)
    for w in range(data.shape[0]):
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                if data[w][i][j][0] > 0:
                    init = out[i][j]
                    out[i][j] = [max(x-100, 0) for x in init]
    img = Image.fromarray(out)
    img.save(PREFIX_DIR + 'overlays/' + name + '.png')

if __name__ == '__main__':
    #data, _ = nrrd.read(PREFIX_DIR + LM_DIR + 'INHS_FISH_51445.nrrd',
    #        index_order='C')
    #mask = gen_mask(data)
    #fdict = gen_dict(mask, 'INHS_FISH_51445')
    #print(fdict)
    #print_image_outline('INHS_FISH_51445')
    #process_nrrds()
    #name = 'INHS_FISH_51445.nrrd'
    #curr_file = PREFIX_DIR + LM_DIR + name
    #data, _ = nrrd.read(curr_file, index_order='C')
    #print_nrrd_map(data, name)
    #DatasetCatalog.register('fish', gen_coco_dataset)
    #MetadataCatalog.get('fish').set(thing_classes=['fish'])
    #gen_coco_dataset()
    #print(f'Saving to file {out_file}')
    #coco.convert_to_coco_json('fish', out_file)
    cfg = get_cfg()
    in_file = PREFIX_DIR + 'fish_val.json'
    register_coco_instances('fish', {}, in_file, PREFIX_DIR + IMAGES_DIR)
    cfg.DATASETS.TRAIN = ('fish',)
    trainer = DefaultTrainer(cfg)
    torch.cuda.empty_cache()
    trainer.train()
