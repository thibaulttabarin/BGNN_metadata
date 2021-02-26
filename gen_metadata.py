import math
import json
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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    predictor = DefaultPredictor(cfg)
    im = cv2.imread(file_path)
    im_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
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
    fish = insts[insts.pred_classes==0]
    if len(fish):
        results['fish'] = [{}]
    else:
        fish = None
    results['has_fish'] = bool(fish)
    try:
        ruler = insts[insts.pred_classes==1][0]
        ruler_bbox = list(ruler.pred_boxes.tensor.cpu().numpy()[0])
        results['ruler_bbox'] = [round(x) for x in ruler_bbox]
    except:
        ruler = None
    results['has_ruler'] = bool(ruler)
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
    visualizer = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
    vis = visualizer.draw_instance_predictions(insts[selector].to('cpu'))
    os.makedirs('images', exist_ok=True)
    file_name = file_path.split('/')[-1]
    print(file_name)
    cv2.imwrite(f'images/gen_mask_prediction_{file_name}',
            vis.get_image()[:, :, ::-1])
    if fish:
        try:
            eyes = insts[insts.pred_classes==2]
        except:
            eyes = None
        for i in range(len(fish)):
            curr_fish = fish[i]
            if eyes:
                eye_ols = [overlap(curr_fish, eyes[j]) for j in
                        range(len(eyes))]
                # TODO: Add pred score as a secondary key in event there are
                #       more than one 1.0 overlap eyes
                max_ind = max(range(len(eye_ols)), key=eye_ols.__getitem__)
                eye = eyes[max_ind]
            else:
                eye = None
            results['fish'][i]['has_eye'] = bool(eye)
            results['fish_count'] = len(insts[(insts.pred_classes==0).
                logical_and(insts.scores > 0.3)])

            bbox_d = [round(x) for x in curr_fish.pred_boxes.tensor.cpu().
                    numpy().astype('float64')[0]]
            im_crop = im_gray[bbox_d[1]:bbox_d[3],bbox_d[0]:bbox_d[2]]
            f_bbox_crop = curr_fish.pred_masks[0].cpu().numpy()\
                    [bbox_d[1]:bbox_d[3],bbox_d[0]:bbox_d[2]]
            fground = im_crop.reshape(-1)[f_bbox_crop.reshape(-1)]
            bground = im_crop.reshape(-1)[np.invert(f_bbox_crop.reshape(-1))]
            sign = -1 if np.mean(bground) > np.mean(fground) else 1
            try:
                bbox, mask = gen_mask(bbox_d, file_path, file_name,
                        val=np.mean(bground) + sign * np.std(bground) * 2)
            except:
                return {file_name: {'errored': True}}
            bbox_d = bbox
            im_crop = im_gray[bbox_d[1]:bbox_d[3],bbox_d[0]:bbox_d[2]]
            f_bbox_crop = curr_fish.pred_masks[0].cpu().numpy()\
                    [bbox_d[1]:bbox_d[3],bbox_d[0]:bbox_d[2]]
            fground = im_crop.reshape(-1)[f_bbox_crop.reshape(-1)]
            bground = im_crop.reshape(-1)[np.invert(f_bbox_crop.reshape(-1))]
            results['fish'][i]['foreground'] = {}
            results['fish'][i]['foreground']['mean'] = np.mean(fground)
            results['fish'][i]['foreground']['std'] = np.std(fground)
            results['fish'][i]['background'] = {}
            results['fish'][i]['background']['mean'] = np.mean(bground)
            results['fish'][i]['background']['std'] = np.std(bground)
            results['fish'][i]['bbox'] = list(bbox)
            #results['fish'][i]['mask'] = mask.astype('uint8').tolist()
            results['fish'][i]['mask'] = '[...]'

            centroid, evec = pca(mask)
            if scale:
                results['fish'][i]['length'] = fish_length(mask, centroid,
                        evec, scale)
            results['fish'][i]['centroid'] = list(centroid)
            if eye:
                #print(fish)
                #print(overlap(fish[i], eye))
                #exit(0)
                eye_center = [round(x) for x in
                        eye.pred_boxes.get_centers()[i].cpu().numpy()]
                results['fish'][i]['eye_center'] = list(eye_center)
                dist1 = distance(centroid, eye_center + evec)
                dist2 = distance(centroid, eye_center - evec)
                if dist2 > dist1:
                    evec *= -1
                if evec[0] <= 0.0:
                    results['fish'][i]['side'] = 'left'
                else:
                    results['fish'][i]['side'] = 'right'
                results['fish'][i]['clock_value'] = clock_value(evec)
            results['fish'][i]['primary_axis'] = list(evec)
    #pprint.pprint(results)
    return {file_name: results}

def angle(vec1, vec2):
    return math.acos(vec1.dot(vec2) / (np.linalg.norm(vec1) *
        np.linalg.norm(vec2)))

def clock_value(evec):
    if evec[0] < 0:
        if evec[1] < 0:
            comp = np.array([0,-1])
            start = 6
        else:
            comp = np.array([-1,0])
            start = 9
    else:
        if evec[1] < 0:
            comp = np.array([1,0])
            start = 3
        else:
            comp = np.array([0,0])
            start = 0
    ang = angle(comp, evec)
    clock = start + ang
    if clock > 11.5:
        clock = 12
    elif clock < 0.5:
        clock = 12
    elif clock >= 0.5 and clock < 1.0:
        clock = 1
    return round(clock)

def fish_length(mask, centroid, evec, scale):
    m1 = evec[1] / evec[0]
    m2 = evec[0] / evec[1]
    x1 = centroid[0]
    y1 = centroid[1]
    x_min = centroid[0]
    x_max = centroid[0]
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            if mask[y,x]:
                x2 = x
                y2 = y
                x_calc = (-y1+y2+m1*x1-m2*x2)/(m1-m2)
                y_calc = m1*(x-x1)+y1
                if x_calc > x_max:
                    x_max = x_calc
                    y_max = y_calc
                elif x_calc < x_min:
                    x_min = x_calc
                    y_min = y_calc
    return distance((x_max, y_max), (x_min, y_min)) / scale

def overlap(fish, eye):
    fish = list(fish.pred_boxes.tensor.cpu().numpy()[0])
    eye = list(eye.pred_boxes.tensor.cpu().numpy()[0])
    if not (fish[0] < eye[2] and eye[0] < fish[2] and fish[1] < eye[3]
            and eye[1] < eye[3]):
        return 0
    pairs = list(zip(fish, eye))
    ol_area = (max(pairs[0]) - min(pairs[2])) * (max(pairs[1]) - min(pairs[3]))
    ol_pct = ol_area / ((eye[0] - eye[2]) * (eye[1] - eye[3]))
    return ol_pct

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

def gen_mask(bbox, file_path, file_name, val=None):
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

        if val is None:
            val = filters.threshold_otsu(arr0) * 1.19
            #print(val)
        #exit(0)
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
                for i in (0,temp.shape[0]-1):
                    for j in (0,temp.shape[1]-1):
                        temp = flood_fill(temp, (i, j), 2)
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

def main():
    direct = sys.argv[1]
    files = [entry.path for entry in os.scandir(direct)]
    print(files)
    with Pool(2) as p:
        results = p.map(gen_metadata, files)
    #results = map(gen_metadata, files)
    output = {}
    for i in results:
        output[list(i.keys())[0]]: list(i.values())[0]
    with open('metadata.json', 'w') as f:
        json.dump(output, f)

if __name__ == '__main__':
    #gen_metadata(sys.argv[1])
    main()
