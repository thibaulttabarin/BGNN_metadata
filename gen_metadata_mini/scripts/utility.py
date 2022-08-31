#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#' Created on Mon Aug  8 11:35:50 2022
#' @author
#' Joel Pepper: initial code
#' Kevin Karnani: modified it
#' Thibault Tabarin: modularize it and trimmed the code for minnow project
#' @description
#' Collect all the code that uses more standard image analyis method


import cv2
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt
from scipy import stats
from skimage import filters, measure
from skimage.morphology import flood_fill, reconstruction
from PIL import Image, ImageDraw

def enhance_contrast(image_arr):
    '''
    Contrast enhance method CLAHE to imporve deep learning prediction

    Parameters
    ----------
    image_arr : np.ndarray (dtype:uint8)
        image to enhance.

    Returns
    -------
    im_enhance : np.ndarray (dtype:uint8)
        CLAHE enhance converted to RGB Color.
    im_gray : np.ndarray (dtype:uint8)
         CLAHE enhance converted to GRAY Color.

    '''
    
    im_gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image_arr, cv2.COLOR_BGR2LAB)
    
    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    im_enhance = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    im_gray = clahe.apply(im_gray)
    
    return im_enhance, im_gray


def calculate_scale(two, three, file_name):
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
    unit = "cm"
    if any(name in file_name for name in in_list):
        scale /= 2.54
        unit = "cm converted"
    elif any(name in file_name for name in cm_list):
        pass
    else:
        unit = "unknown"
    return scale, unit

def distance(pt1, pt2):
    """
    Returns the 2-D Euclidean Distance between 2 points.
    """
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)  

def adaptive_threshold(bbox, im_gray, VAL_SCALE_FAC):
    """
    Determines the best thresholding value.
    Parameters
    ----------
    bbox : list (int)
        Bounding box in [left, top, right, bottom] format.
    im_gray : np.ndarray 
        Grayscale version of original image.
    VAL_SCALE_FAC : int
        value scale factor.

    Returns
    -------
    val: float 
        new threshold

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

def generate_pixel_analysis(bbox, im_gray, VAL_SCALE_FAC, detectron_mask, flipped=False):
    """
    Generates a new mask for the fish using the puxel analysis.
    Floodfills to make a whole image.
    bbox : round up bounding box from detectron
    val : adaptative threshold from fun adaptive_threshold
    detectron_mask : mask from detectron output
    """
    
    val = adaptive_threshold(bbox, im_gray, VAL_SCALE_FAC)
    failed = False
    bbox_orig = bbox.copy()
    left, top, right, bottom = bbox
    
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
                    print('ERROR on flood fill')
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
            print('Error expanding bounding box')
            failed = True
            return bbox_orig, detectron_mask.astype('uint8'), failed
        # New bbox
        bbox = [left, top, right, bottom]
        # New threshold
        val = adaptive_threshold(bbox, im_gray, VAL_SCALE_FAC)
    if np.count_nonzero(thresh) / im_crop.size < .1:
        
        new_mask = detectron_mask.astype('uint8')
        bbox = bbox_orig
        failed = True
    return bbox, new_mask, failed

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


def clean_regionprop(mask):
    '''
    Create a clean regionprop for morphology analysis
    Fill the teh hole in the mask image

    Parameters
    ----------
    mask : np.ndarray 
        DESCRIPTION.

    Returns
    -------
    filled : regionprop instance
        DESCRIPTION.

    '''
    
    # clean hole in the mask image    
    seed = np.copy(mask)
    seed[1:-1, 1:-1] = mask.max()
    filled = reconstruction(seed, mask, method='erosion')
    
    # Create the region prop
    mask_label = measure.label(filled)
    mask_region = measure.regionprops(mask_label)
    
    # get the biggest blod (region)
    biggest_region = sorted(mask_region, key=lambda r: r.area, reverse=True)[0]
    
    return biggest_region

def ioa_(box1, box2):
    '''
    Intersection area over teh smallest area correspond to the percent of overlap between 2 bounding box (rectangle)

    Parameters
    ----------
    box1: list [left, top, right, bottom]
        biggest bounding box.
    box2 : list [left, top, right, bottom]
        smallest bounding box.

    Returns
    -------
    float
        Intersection over area ioa .

    '''
    
    lt = max(box1[:2],box2[:2]) # left top of the intersection
    rb = min(box1[2:],box2[2:]) # right bottom of the intersection
    
    width_height = np.subtract(rb,lt).clip(min=0) # width heig of the intersection
    intersection_area = width_height[0]*width_height[1]
    
    # Sanity check get the smallest box
    area_bbox1 = (box1[0]-box1[2])*(box1[1]-box1[3])
    area_bbox2 = (box2[0]-box2[2])*(box2[1]-box2[3])
    min_area = min(area_bbox1, area_bbox2)
    
    return intersection_area/min_area


def get_morphological_value(mask):
    '''
    Calculate the morphological and Statistical information from an image mask.
    
    Parameters
    ----------
    mask : np.ndarray (binary: 0,1)
        Mask of the fish .

    Returns
    -------
    dict_Morpho_info : dictionnary
        {'extent':'None', 'eccentricity':'None','solidity':'None', 'skew':'None', 
         'kurtosis':'None'}.

    '''
    
    dict_Morpho_info={'extent':'None', 'eccentricity':'None',
                      'solidity':'None', 'skew':'None', 'kurtosis':'None'}
    
    # Create a clean regionprop for morphology analysis of the mask
    region = clean_regionprop(mask)

    # Morphological infarmation
    dict_Morpho_info['extent'] = region.extent
    dict_Morpho_info['eccentricity'] = region.eccentricity
    dict_Morpho_info['solidity'] = region.solidity
    
    # Statistic information
    mask_coords = np.argwhere(mask != 0)[:, [1, 0]]
    dict_Morpho_info['skew'] = list(stats.skew(mask_coords))
    dict_Morpho_info['kurtosis'] = list(stats.kurtosis(mask_coords))

    return dict_Morpho_info

def get_brightness(im, mask, bbox):
    '''
    Get quality image information such as foreground the fish and background around teh fish in the
    bounding box.

    Parameters
    ----------
    im : np.ndarray
        original image.
    mask : np.ndarray np.uint8
        mask of the fish.
    bbox : list of int
        bounding box around the fish.

    Returns
    -------
    dict_image_quality : dictionnary
        information on image quality, background an aforeground (fish).

    '''
    
    dict_brightness = {'foreground_mean':'None', 'foreground_std':'None', 
                          'background_mean':'None', 'background_std':'None'}
    
    _, im_gray = enhance_contrast(im)
    im_crop = im_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]].reshape(-1)
    mask_crop = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].reshape(-1)
    fground = im_crop[np.where(mask_crop)]
    bground = im_crop[np.where(np.logical_not(mask_crop))]
    
    dict_brightness['foreground_mean'] = round(np.mean(fground),2)
    dict_brightness['foreground_std'] = round(np.std(fground),2)

    dict_brightness['background_mean'] = round(np.mean(bground),2)
    dict_brightness['background_std'] = round(np.std(bground),2)

    return dict_brightness


def check_bbox (im, dict_fish, center=None):
    '''
    Visualization tools to assess correctness of the result.
    Plot on the original image the bounding box for the fish and the eye and
    if provided the center (of the eye).

    Parameters
    ----------
    im : np.ndarray
        image to visualize.
    dict_fish : dictionnary
        Dictionnary containing metadata for fish
    center : list (int), optional
        Coordinate of center for example eye. The default is None.

    Returns
    -------
    img : np.ndarray
        image .

    '''
    
    img = Image.fromarray(im)
    img1 = ImageDraw.Draw(img)
    
    left,top,right,bottom = dict_fish['bbox']
    shape_fish = [(left, top), (right,bottom)]
    img1.rectangle(shape_fish, outline ="red")
    
    left,top,right,bottom = dict_fish['eye_bbox']
    shape_eye = [(left, top), (right,bottom)]
    img1.rectangle(shape_eye, outline ="red")
    
    if center :
        row,col = center
        xy = [(col-9,row-9),(col+9,row+9)]
        img1.ellipse(xy, fill='gray', outline=None, width=1)
        
    return img
    