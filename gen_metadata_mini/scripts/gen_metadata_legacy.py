#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This is Kevin Karnani code, modified by thibault tabarin for containerization

'''
import json
import math
import os
import pprint
import sys
import yaml
from random import shuffle

import gc
import torch
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.data import Metadata
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Boxes, pairwise_iou, pairwise_ioa
from matplotlib import pyplot as plt
from scipy import stats
from skimage import filters, measure
from skimage.morphology import flood_fill
from torch.multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore")
# torch.multiprocessing.set_start_method('forkserver')

# ensure the look at the right place for the configuration file
# setting path
try:
    root_file_path = os.path.dirname(__file__)
except:
    root_file_path = './'

# import configuration
conf = json.load(open(os.path.join(root_file_path,'config/config.json'), 'r'))
PROCESSOR = conf['PROCESSOR']
MODEL_WEIGHT = os.path.join(root_file_path, conf['MODEL_WEIGHT'])
NUM_CLASSES = conf['NUM_CLASSES']
VAL_SCALE_FAC = conf['VAL_SCALE_FAC']
IOU_PCT = conf['IOU_PCT']
ENHANCE = conf['ENHANCE']

"output/model_final.pth"

with open(os.path.join(root_file_path,'config/mask_rcnn_R_50_FPN_3x.yaml'), 'r') as f:
    iters = yaml.load(f, Loader=yaml.FullLoader)["SOLVER"]["MAX_ITER"]

def init_model(processor=PROCESSOR, model_weight=MODEL_WEIGHT):
    """
    Initialize model using config files for RCNN, the trained weights, and other parameters.

    Returns:
        predictor -- DefaultPredictor(**configs).
    """
    root_file_path = os.path.dirname(__file__)
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(root_file_path,'config/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    OUTPUT_DIR = os.path.join(root_file_path, 'output')

    #
    cfg.MODEL.WEIGHTS = model_weight
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.DEVICE = processor
    predictor = DefaultPredictor(cfg)

    return predictor


def gen_metadata(file_path, enhance_contrast=ENHANCE, visualize=False, multiple_fish=False):
    """
    Generates metadata of an image and stores attributes into a Dictionary.

    Parameters:
        file_path -- string of path to image file.
    Returns:
        {file_name: results} -- dictionary of file and associated results.

    """
    # Extract file name base
    file_name = file_path.split('/')[-1]
    f_name = file_name.split('.')[0]

    # Initialize model
    predictor = init_model()
    im = cv2.imread(file_path)
    im_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if enhance_contrast:
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)

        # -----Splitting the LAB image to different channels-------------------------
        l, a, b = cv2.split(lab)

        # -----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl, a, b))

        # -----Converting image from LAB Color model to RGB model--------------------
        im = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        im_gray = clahe.apply(im_gray)
    metadata = Metadata(evaluator_type='coco', image_root='.',
                        json_file='',
                        name='metadata',
                        thing_classes=['fish', 'ruler', 'eye', 'two', 'three'],
                        thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
                        )
    output = predictor(im)
    insts = output['instances']

    results = {}
    #results['tag']='friday'

    fish = insts[insts.pred_classes == 0]
    if len(fish):
        results['fish'] = []
        results['fish'].append({})
    else:
        fish = None

    results['has_fish'] = bool(fish)
    try:
        ruler = insts[insts.pred_classes == 1][0]
        ruler_bbox = list(ruler.pred_boxes.tensor.cpu().numpy()[0])
        results['ruler_bbox'] = [round(x) for x in ruler_bbox]
    except:
        ruler = None
    results['has_ruler'] = bool(ruler)
    try:
        two = insts[insts.pred_classes == 3][0]
    except:
        two = None
    try:
        three = insts[insts.pred_classes == 4][0]
    except:
        three = None
    if ruler and two and three:
        scale = calc_scale(two, three, file_name)
        results['scale'] = scale
        results['unit'] = 'cm'
    else:
        scale = None

    skippable_fish = []
    fish_length = 0
    if fish:
        try:
            eyes = insts[insts.pred_classes == 2]
        except:
            eyes = None

        fish = fish[fish.scores > .3]
        fish_length = len(fish)
        fish = fish[fish.scores.argmax().item()]

        for i in range(len(fish)):
            curr_fish = fish[i]

            if eyes:
                eye_ols = [overlap(curr_fish, eyes[j]) for j in
                           range(len(eyes))]


                eye = None
                if not all(ol == 0 for ol in eye_ols):
                    full = [i for i in range(
                        len(eye_ols)) if eye_ols[i] >= .95]

                    # if multiple eyes with 95% or greater overlap, pick highest confidence
                    if len(full) > 1:
                        eye = eyes[full]
                        eye = eye[eye.scores.argmax().item()]

                    else:
                        max_ind = max(range(len(eye_ols)),
                                      key=eye_ols.__getitem__)
                        eye = eyes[max_ind]
            else:
                eye = None

            bbox = [round(x) for x in curr_fish.pred_boxes.tensor.cpu().numpy().astype('float64')[0]]
            need_scaling = False
            detectron_mask = curr_fish.pred_masks[0].cpu().numpy()
            val = adaptive_threshold(bbox, im_gray)
            bbox, mask, pixel_anal_failed = gen_mask(bbox, file_path,
                                                     file_name, im_gray, val, detectron_mask)

            mask_uint8 = np.where(mask == 1, 255, 0).astype(np.uint8)

            centroid, evecs, cont_length, cont_width, length, width, area = pca(mask, scale)
            major, minor = evecs[0], evecs[1]

            if not np.count_nonzero(mask):
                print('Mask failed: {file_name}')
                results['errored'] = True
            else:
                im_crop = im_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]].reshape(-1)
                mask_crop = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].reshape(-1)
                mask_coords = np.argwhere(mask != 0)[:, [1, 0]]
                fground = im_crop[np.where(mask_crop)]
                bground = im_crop[np.where(np.logical_not(mask_crop))]
                results['fish'][i]['foreground'] = {}
                results['fish'][i]['foreground']['mean'] = np.mean(fground)
                results['fish'][i]['foreground']['std'] = np.std(fground)
                results['fish'][i]['background'] = {}
                results['fish'][i]['background']['mean'] = np.mean(bground)
                results['fish'][i]['background']['std'] = np.std(bground)
                results['fish'][i]['bbox'] = list(bbox)
                results['fish'][i]['pixel_analysis_failed'] = pixel_anal_failed
                #start, code = encoded_mask(mask)
                region = measure.regionprops(mask)[0]

                results['fish'][i]['extent'] = region.extent
                results['fish'][i]['eccentricity'] = region.eccentricity
                results['fish'][i]['solidity'] = region.solidity
                results['fish'][i]['skew'] = list(stats.skew(mask_coords))
                results['fish'][i]['kurtosis'] = list(
                    stats.kurtosis(mask_coords))
                results['fish'][i]['std'] = list(np.std(mask_coords, axis=0))
                #results['fish'][i]['mask'] = {}
                #results['fish'][i]['mask']['start_coord'] = list(start)
                #results['fish'][i]['mask']['encoding'] = code
                results['fish'][i]['rescale'] = 'no'

                # upscale fish and then rerun
                if eye is None:
                    need_scaling = True
                    factor = 4
                    eye_center, side, clock_val = upscale(
                        im, bbox, f_name, factor)
                    results['fish'][i]['rescale'] = 'yes'
                    if eye_center is not None and side is not None:
                        results['fish'][i]['eye_center'] = eye_center
                        results['fish'][i]['side'] = side
                        results['fish'][i]['clock_value'] = clock_val
                        eye = 1  # placeholder, change to something more useful
                if scale:
                    results['fish'][i]['cont_length'] = cont_length
                    results['fish'][i]['cont_width'] = cont_width
                    results['fish'][i]['area'] = area
                    results['fish'][i]['feret_diameter_max'] = region.feret_diameter_max / scale
                    results['fish'][i]['major_axis_length'] = region.major_axis_length / scale
                    results['fish'][i]['minor_axis_length'] = region.minor_axis_length / scale
                    results['fish'][i]['convex_area'] = region.convex_area / \
                                                        (scale ** 2)
                    results['fish'][i]['perimeter'] = measure.perimeter(
                        mask, neighbourhood=8) / scale
                    results['fish'][i]['oriented_length'] = length / scale
                    results['fish'][i]['oriented_width'] = width / scale
                results['fish'][i]['centroid'] = centroid.tolist()
            results['fish'][i]['has_eye'] = bool(eye)

            if eye and not need_scaling:
                eye_center = [round(x) for x in eye.pred_boxes.get_centers()[0].cpu().numpy()]
                results['fish'][i]['eye_center'] = list(eye_center)
                dist1 = distance(centroid, eye_center + major)
                dist2 = distance(centroid, eye_center - major)
                if dist2 > dist1:
                    major *= -1
                if major[0] <= 0.0:
                    results['fish'][i]['side'] = 'left'
                else:
                    results['fish'][i]['side'] = 'right'
                snout_vec = major
                if snout_vec is None:
                    results['fish'][i]['clock_value'] = \
                        clock_value(major, file_name)
                else:
                    results['fish'][i]['clock_value'] = \
                        clock_value(snout_vec, file_name)
                results['fish'][i]['primary_axis'] = list(major)
                results['fish'][i]['score'] = float(curr_fish.scores[0].cpu())
    results['fish_count'] = len(insts[(insts.pred_classes == 0).logical_and(insts.scores > 0.3)]) - \
                            len(skippable_fish) if multiple_fish else int(results['has_fish'])
    results['detected_fish_count'] = fish_length
    return {f_name: results}, mask_uint8


def gen_metadata_upscale(file_path, fish):
    gc.collect()
    torch.cuda.empty_cache()
    predictor = init_model()
    im = fish
    im_gray = cv2.cvtColor(fish, cv2.COLOR_BGR2GRAY)
    output = predictor(im)
    insts = output['instances']

    results = {}
    file_name = file_path.split('/')[-1]
    f_name = file_name.split('.')[0]

    fish = insts[insts.pred_classes == 0]
    if len(fish):
        results['fish'] = []
        results['fish'].append({})
    else:
        fish = None
    results['has_fish'] = bool(fish)
    if fish:
        try:
            eyes = insts[insts.pred_classes == 2]
        except:
            eyes = None

        fish = fish[fish.scores > .3]
        fish = fish[fish.scores.argmax().item()]
        for i in range(len(fish)):
            curr_fish = fish[i]
            if eyes:
                eye_ols = [overlap(curr_fish, eyes[j]) for j in
                           range(len(eyes))]
                eye = None
                if not all(ol == 0 for ol in eye_ols):
                    full = [i for i in range(
                        len(eye_ols)) if eye_ols[i] >= .95]

                    # if multiple eyes with 95% or greater overlap, pick highest confidence
                    if len(full) > 1:
                        eye = eyes[full]
                        eye = eye[eye.scores.argmax().item()]
                    else:
                        max_ind = max(range(len(eye_ols)),
                                      key=eye_ols.__getitem__)
                        eye = eyes[max_ind]
            else:
                eye = None
            bbox = [round(x) for x in curr_fish.pred_boxes.tensor.cpu().numpy().astype('float64')[0]]
            detectron_mask = curr_fish.pred_masks[0].cpu().numpy()
            val = adaptive_threshold(bbox, im_gray)
            bbox, mask, pixel_anal_failed = gen_mask_upscale(bbox, file_path,
                                                             file_name, im_gray, val, detectron_mask)
            centroid, evecs = pca(mask)[:2]
            major, minor = evecs[0], evecs[1]
            results['fish'][i]['has_eye'] = bool(eye)
            if eye:
                eye_center = [round(x) for x in eye.pred_boxes.get_centers()[0].cpu().numpy()]
                results['fish'][i]['eye_center'] = list(eye_center)
                dist1 = distance(centroid, eye_center + major)
                dist2 = distance(centroid, eye_center - major)
                if dist2 > dist1:
                    major *= -1
                results['fish'][i]['side'] = 'left' if major[0] <= 0.0 else 'right'
                snout_vec = major
                results['fish'][i]['clock_value'] = clock_value(major if snout_vec is None else snout_vec, file_name)
    return {f_name: results}


def upscale(im, bbox, f_name, factor):
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    scaled = cv2.resize(im[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy(), (w * factor, h * factor),
                        interpolation=cv2.INTER_CUBIC)
    os.makedirs('images/testing', exist_ok=True)
    cv2.imwrite(f'images/testing/{f_name}.png', scaled)
    eye_center, side, clock_val, scale = None, None, None, None
    new_data = gen_metadata_upscale(f'images/testing/{f_name}.png', scaled)
    if 'fish' in new_data[f'{f_name}'] and new_data[f'{f_name}']['fish'][0]['has_eye']:
        eye_center = new_data[f'{f_name}']['fish'][0]['eye_center']
        eye_x, eye_y = eye_center
        eye_y //= factor
        eye_y += bbox[1]
        eye_x //= factor
        eye_x += bbox[0]
        eye_center = [eye_x, eye_y]
        side = new_data[f'{f_name}']['fish'][0]['side']
        clock_val = new_data[f'{f_name}']['fish'][0]['clock_value']
    if os.path.isfile(f'images/testing/{f_name}.png'):
        os.remove(f'images/testing/{f_name}.png')
    return eye_center, side, clock_val


def adaptive_threshold(bbox, im_gray):
    """
    Determines the best thresholding value.
    Parameters
    ----------
        bbox: list (int)
        bounding box in [left, top, right, bottom] format.
        im_gray: np.ndarray
        grayscale version of original image.
    Returns:
        val: int
        new threshold.
    """
    im_crop = im_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    val = filters.threshold_otsu(im_crop)
    mask = np.where(im_crop > val, 1, 0).astype(np.uint8)
    flat_mask = mask.reshape(-1)
    bground = im_crop.reshape(-1)[np.where(np.logical_not(flat_mask))]
    mean_b = np.mean(bground)
    flipped = False
    diff = abs(mean_b - val)
    if flipped:
        val -= diff * VAL_SCALE_FAC
    else:
        val += diff * VAL_SCALE_FAC
    val = min(max(1, val), 254)
    return val


def angle(vec1, vec2):
    """
    Finds angle between two vectors.
    """
    # print(f'angle: {vec1}, {vec2}')
    return math.acos(vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def clock_value(evec, file_name):
    """
    Creates a clock value depending on the major axis provided.
    Parameters:
        evec -- Eigenvector that depicts the major axis.
        file_name -- path to image file.
    Returns:
        round(clock) -- rounded off clock value, ranging from 1-12.
    """
    if evec[0] < 0:
        if evec[1] > 0:
            comp = np.array([-1, 0])
            start = 9
        else:
            comp = np.array([0, -1])
            start = 6
    else:
        if evec[1] < 0:
            comp = np.array([1, 0])
            start = 3
        else:
            comp = np.array([0, 1])
            start = 0
    ang = angle(comp, evec)
    clock = start + (ang / (2 * math.pi) * 12)
    if clock > 11.5:
        clock = 12
    elif clock < 0.5:
        clock = 12
    return round(clock)



def overlap(fish, eye):
    """
    Checks if the eye is in the fish.
    Parameters:
        fish -- fish coordinates.
        eye -- eye coordinates.
    Returns:
        ol_pct -- percent of eye that is inside the fish.
    """
    fish = list(fish.pred_boxes.tensor.cpu().numpy()[0])
    eye = list(eye.pred_boxes.tensor.cpu().numpy()[0])
    if not (fish[0] < eye[2] and eye[0] < fish[2] and fish[1] < eye[3]
            and eye[1] < eye[3]):
        return 0
    pairs = list(zip(fish, eye))
    ol_area = (max(pairs[0]) - min(pairs[2])) * (max(pairs[1]) - min(pairs[3]))
    ol_pct = ol_area / ((eye[0] - eye[2]) * (eye[1] - eye[3]))
    return ol_pct



# https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
def pca(img, glob_scale=None, visualize=False):
    """
    Performs principle component analysis on a grayscale image.
    Parameters:
        img -- grayscale image.
        glob_scale -- pixels per unit.
    Returns:
        np.array(centroid) -- numpy array containing centroid.
        evecs[:, sort_indices[0]] -- major axis, or eigenvector associated with highest eigenvalue.
        length -- length of fish.
        width -- width of fish.
        area -- area of fish.
    """
    moments = cv2.moments(img)
    centroid = (int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"]))
    y, x = np.nonzero(img)

    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])

    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    # Eigenvector with largest eigenvalue
    x_v1, y_v1 = evecs[:, sort_indices[0]]
    # negate eigenvector
    if x_v1 < 0:
        x_v1 *= -1
        y_v1 *= -1
    theta = np.arctan2(y_v1, x_v1)
    rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
    transformed_mat = rotation_mat * coords
    x_transformed, y_transformed = transformed_mat.A
    x_round, y_round = x_transformed.round(
        decimals=0), y_transformed.round(decimals=0)
    x_vals, x_counts = np.unique(x_round, return_counts=True)
    y_vals, y_counts = np.unique(y_round, return_counts=True)
    x_calc, y_calc = x_vals[x_counts.argmax()], y_vals[y_counts.argmax()]
    x_indices, y_indices = np.where(
        x_round == x_calc), np.where(y_round == y_calc)
    cont_width = y_round[x_indices].max() - y_round[x_indices].min()
    cont_length = x_round[y_indices].max() - x_round[y_indices].min()
    width = y_vals.max() - y_vals.min()
    length = x_vals.max() - x_vals.min()

    if visualize:
        x_v2, y_v2 = evecs[:, sort_indices[1]]
        scale = 300
        plt.plot([x_v1 * -scale * 2, x_v1 * scale * 2],
                 [y_v1 * -scale * 2, y_v1 * scale * 2], color='red')
        plt.plot([x_v2 * -scale, x_v2 * scale],
                 [y_v2 * -scale, y_v2 * scale], color='blue')
        plt.plot(x, y, 'y.')
        plt.axis('equal')
        plt.gca().invert_yaxis()  # Match the image system with origin at top left
        plt.axhline(y=y_calc)
        plt.axvline(x=x_calc)
        plt.plot(x_transformed, y_transformed, 'g.')
        plt.show()

    area = transformed_mat.shape[1]
    if glob_scale is not None:
        cont_length /= glob_scale
        cont_width /= glob_scale
        length /= glob_scale
        width /= glob_scale
        area /= glob_scale ** 2

    return np.array(centroid), evecs[:, sort_indices], cont_length, cont_width, length, width, area




def distance(pt1, pt2):
    """
    Returns the 2-D Euclidean Distance between 2 points.
    """
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def calc_scale(two, three, file_name):
    """
    Calculates the pixels per unit.
    Parameters:
        two -- the "two" from the ruler in the image.
        three -- the "three" from the ruler in the image.
        file_name -- name of Image in file path.
    Returns:
        scale -- pixels between the centers of the "two" and "three".
    """
    cm_list = ['uwzm']
    in_list = ['inhs']
    file_name = file_name.lower()
    pt1 = two.pred_boxes.get_centers()[0]
    pt2 = three.pred_boxes.get_centers()[0]
    scale = distance([float(pt1[0]), float(pt1[1])],
                     [float(pt2[0]), float(pt2[1])])
    if any(name in file_name for name in in_list):
        scale /= 2.54
    elif any(name in file_name for name in cm_list):
        pass
    else:
        scale /= 2.54
        print("Unable to determine unit. Defaulting to cm.")
    return scale


def gen_mask(bbox, file_path, file_name, im_gray, val, detectron_mask, flipped=False):
    """
    Generates the mask for the fish and floodfills to make a whole image.
    """
    failed = False
    left = round(bbox[0])
    right = round(bbox[2])
    top = round(bbox[1])
    bottom = round(bbox[3])
    bbox_orig = bbox
    bbox = (left, top, right, bottom)

    im = im_gray.copy()
    shape = im.shape
    done = False
    im_crop = im[top:bottom, left:right]
    fish_pix, thresh, new_mask = None, None, None

    while not done:
        done = True
        im_crop = im[top:bottom, left:right]
        count = 0
        thresh = np.where(im_crop < val, 1, 0).astype(np.uint8)
        indices = list(zip(*np.where(thresh == 1)))
        shuffle(indices)
        for ind in indices:
            if fish_pix is not None:
                ind = fish_pix
            count += 1
            # if 10k pass and fish not found
            if count > 10000:
                if fish_pix is not None:
                    fish_pix = None
                else:
                    print(f'ERROR on flood fill: {file_name}')
                    return bbox_orig, detectron_mask.astype('uint8'), True
            temp = flood_fill(thresh, ind, 2)
            temp = np.where(temp == 2, 1, 0)
            percent = np.count_nonzero(temp) / im_crop.size
            if percent > 0.1:
                fish_pix = ind
                for i in (0, temp.shape[0] - 1):
                    for j in (0, temp.shape[1] - 1):
                        temp = flood_fill(temp, (i, j), 2)
                thresh = np.where(temp != 2, 1, 0).astype(np.uint8)
                break
        new_mask = np.full(shape, 0).astype(np.uint8)
        new_mask[top:bottom, left:right] = thresh
        # Expands the bounding box
        try:
            if np.any(new_mask[top:bottom, left] != 0) and left > 0:
                left -= 1
                left = max(0, left)
                done = False
            if np.any(new_mask[top:bottom, right] != 0) and right < shape[1] - 1:
                right += 1
                right = min(shape[1] - 1, right)
                done = False
            if np.any(new_mask[top, left:right] != 0) and top > 0:
                top -= 1
                top = max(0, top)
                done = False
            if np.any(new_mask[bottom, left:right] != 0) and bottom < shape[0] - 1:
                bottom += 1
                bottom = min(shape[0] - 1, bottom)
                done = False
        except:
            print(f'{file_name}: Error expanding bounding box')
            # done = True
            return bbox_orig, detectron_mask.astype('uint8'), True
        # New bbox
        bbox = (left, top, right, bottom)
        # New threshold
        val = adaptive_threshold(bbox, im_gray)
    if np.count_nonzero(thresh) / im_crop.size < .1:
        print(f'{file_name}: Using detectron mask and bbox')
        new_mask = detectron_mask.astype('uint8')
        bbox = bbox_orig
        failed = True
    return bbox, new_mask, failed


def gen_mask_upscale(bbox, file_path, file_name, im_gray, val, detectron_mask):
    failed = False
    l = round(bbox[0])
    r = round(bbox[2])
    t = round(bbox[1])
    b = round(bbox[3])
    bbox_orig = bbox
    bbox = (l, t, r, b)

    im = im_gray.copy()
    im_crop = im[t:b, l:r]
    thresh = np.where(im_crop < val, 1, 0).astype(np.uint8)
    new_mask = np.full(im.shape, 0).astype(np.uint8)
    new_mask[t:b, l:r] = thresh
    if np.count_nonzero(thresh) / im_crop.size < .1:
        print(f'{file_name}: Using detectron mask and bbox')
        new_mask = detectron_mask.astype('uint8')
        bbox = bbox_orig
        failed = True
    return bbox, new_mask, failed


def gen_metadata_safe(file_path):
    """
    Deals with erroneous metadata generation errors.
    """
    try:
        result, mask_uint8 = gen_metadata(file_path)
        return result, mask_uint8
    except Exception as e:
        print(f'{file_path}: Errored out ({e})')
        return {file_path: {'errored': True}}


def main(input_file, output_result, output_mask=None):

    result, mask_uint8 = gen_metadata_safe(input_file)

    with open(output_result, 'w') as f:
            json.dump(result, f)

    if output_mask != None:
        cv2.imwrite(output_mask, mask_uint8)

    return print('This is the user version 2.1!')

if __name__ == '__main__':

    if len(sys.argv) > 2: # case with 2 arguments input
        input_file = sys.argv[1]
        output_json = sys.argv[2]
        output_mask = None

    if len(sys.argv) == 4 : # case if there is a 3rd argument input
        output_mask = sys.argv[3]

    main(input_file, output_json, output_mask)
