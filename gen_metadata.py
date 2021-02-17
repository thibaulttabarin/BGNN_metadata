import sys
import os
from multiprocessing import Pool
import pandas as pd
import numpy as np
import nrrd
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt
import pprint

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
from detectron2.engine import DefaultPredictor
from detectron2.data import Metadata
from detectron2.utils.visualizer import Visualizer

import cv2

from skimage import filters
from skimage.morphology import flood_fill
from random import shuffle

def gen_metadata(file_path):
    cfg = get_cfg()
    cfg.merge_from_file("config/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    predictor = DefaultPredictor(cfg)
    im = cv2.imread(file_path)
    metadata = Metadata(evaluator_type='coco', image_root='.',
            json_file='',
            name='metadata',
            thing_classes=['fish', 'ruler', 'eye', 'two', 'three'],
            thing_dataset_id_to_contiguous_id=
                {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
            )
    output = predictor(im)
    insts = output['instances']
    selector = insts.pred_classes==0
    selector = selector.cumsum(axis=0).cumsum(axis=0) == 1
    results = {}
    for i in range(1, 5):
        temp = insts.pred_classes==i
        selector += temp.cumsum(axis=0).cumsum(axis=0) == 1
    try:
        fish = insts[insts.pred_classes==0][0]
        results['fish'] = [{}]
    except:
        fish = None
    results['is_fish'] = bool(fish)
    try:
        ruler = insts[insts.pred_classes==1][0]
        print(ruler)
        ruler_bbox = list(ruler.pred_boxes.tensor.cpu().numpy()[0])
        results['ruler_bbox'] = [round(x) for x in ruler_bbox]
    except:
        ruler = None
    results['is_ruler'] = bool(ruler)
    try:
        two = insts[insts.pred_classes==3][0]
    except:
        two = None
    try:
        three = insts[insts.pred_classes==4][0]
    except:
        three = None
    if ruler and two and three:
        scale = calc_scale(two, three)
        results['scale'] = scale
    if fish:
        try:
            eye = insts[insts.pred_classes==2][0]
        except:
            eye = None
        results['fish'][0]['is_eye'] = bool(eye)
        results['fish_count'] = len(insts[(insts.pred_classes==0).logical_and(
                insts.scores > 0.3)])
        visualizer = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
        vis = visualizer.draw_instance_predictions(insts[selector].to('cpu'))
        os.makedirs('images', exist_ok=True)
        file_name = file_path.split('/')[-1]
        print(file_name)
        cv2.imwrite(f'images/gen_mask_prediction_{file_name}',
                vis.get_image()[:, :, ::-1])

        bbox, mask = gen_mask(
                fish.pred_boxes.tensor.cpu().numpy().astype('float64')[0],
                file_path, file_name)
        results['fish'][0]['bbox'] = list(bbox)
        #results['mask'] = mask.astype('uint8').tolist()
        results['mask'] = '[...]'

        centroid, evec = pca(mask)
        results['fish'][0]['centroid'] = list(centroid)
        if eye:
            eye_center = eye.pred_boxes.get_centers()[0].cpu().numpy()
            results['fish'][0]['eye_center'] = list(eye_center)
            dist1 = distance(centroid, eye_center + evec)
            dist2 = distance(centroid, eye_center - evec)
            if dist2 > dist1:
                evec *= -1
        results['fish'][0]['primary_axis'] = list(evec)
        if evec[0] <= 0.0:
            results['fish'][0]['side'] = 'left'
        else:
            results['fish'][0]['side'] = 'right'
    pprint.pprint(results)


def pca(img):
    moments = cv2.moments(img)
    centroid = (int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]))
    #print(centroid)
    y, x = np.nonzero(img)

    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])

    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)

    sort_indices = np.argsort(evals)[::-1]
    return (np.array(centroid), evecs[:, sort_indices[0]])

def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def calc_scale(two, three):
    pt1 = two.pred_boxes.get_centers()[0]
    pt2 = three.pred_boxes.get_centers()[0]
    scale = distance([float(pt1[0]), float(pt1[1])],
            [float(pt2[0]), float(pt2[1])])
    scale /= 2.54
    #print(f'Pixels/cm: {scale}')
    return scale

def gen_mask(bbox, file_path, file_name):
    l = round(bbox[0])
    r = round(bbox[2])
    t = round(bbox[1])
    b = round(bbox[3])

    im = Image.open(file_path).convert('L')
    arr2 = np.array(im)
    shape = arr2.shape
    done = False
    while not done:
        done = True
        bbox = (l,t,r,b)
        arr0 = np.array(im.crop(bbox))
        bb_size = arr0.size

        val = filters.threshold_otsu(arr0) * 1.19
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
            if percent > 0.1:
                temp = flood_fill(temp, (0, 0), 2)
                arr1 = np.where(temp != 2, 1, 0).astype(np.uint8)
                break
        arr3 = np.full(shape, 0).astype(np.uint8)
        arr3[t:b,l:r] = arr1
        if np.any(arr3[t:b,l] != 0):
            l -= 1
            done = False
        if np.any(arr3[t:b,r] != 0):
            r += 1
            done = False
        if np.any(arr3[t,l:r] != 0):
            t -= 1
            done = False
        if np.any(arr3[b,l:r] != 0):
            b += 1
            done = False
    arr4 = np.where(arr3 == 1, 255, 0).astype(np.uint8)
    (l,t,r,b) = shrink_bbox(arr3)
    arr4[t:b,l] = 175
    arr4[t:b,r] = 175
    arr4[t,l:r] = 175
    arr4[b,l:r] = 175
    im2 = Image.fromarray(arr4, 'L')
    im2.save(f'images/gen_mask_mask_{file_name}')
    return (bbox, arr3)

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
def shrink_bbox(mask):
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

    return (cmin, rmin, cmax, rmax)

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

if __name__ == '__main__':
    gen_metadata(sys.argv[1])
