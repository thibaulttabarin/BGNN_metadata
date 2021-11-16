import json
import math
import os
import pprint
import sys
from random import shuffle

import cv2
import numpy as np
from PIL import Image
from detectron2.config import get_cfg
from detectron2.data import Metadata
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt
from skimage import filters, measure
from skimage.morphology import flood_fill
from torch.multiprocessing import Pool

# torch.multiprocessing.set_start_method('forkserver')

VAL_SCALE_FAC = 0.5


def init_model():
    """
    Initialize model using config files for RCNN, the trained weights, and other parameters.

    Returns:
        predictor -- DefaultPredictor(**configs).
    """
    cfg = get_cfg()
    cfg.merge_from_file("config/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    predictor = DefaultPredictor(cfg)
    return predictor


def gen_metadata(file_path):
    """
    Generates metadata of an image and stores attributes into a Dictionary.

    Parameters:
        file_path -- string of path to image file.
    Returns:
        {file_name: results} -- dictionary of file and associated results.
    """
    predictor = init_model()
    im = cv2.imread(file_path)
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

    im_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    im_gray = clahe.apply(im_gray)
    metadata = Metadata(evaluator_type='coco', image_root='.',
                        json_file='',
                        name='metadata',
                        thing_classes=['fish', 'ruler', 'eye', 'two', 'three'],
                        thing_dataset_id_to_contiguous_id=
                        # {1: 0}
                        {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
                        )
    output = predictor(im)
    insts = output['instances']
    selector = insts.pred_classes == 0
    selector = selector.cumsum(axis=0).cumsum(axis=0) == 1
    results = {}
    file_name = file_path.split('/')[-1]
    for i in range(1, 5):
        temp = insts.pred_classes == i
        selector += temp.cumsum(axis=0).cumsum(axis=0) == 1
    fish = insts[insts.pred_classes == 0]
    # print(fish)
    if len(fish):
        results['fish'] = []
        for _ in range(len(fish)):
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
    visualizer = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
    # vis = visualizer.draw_instance_predictions(insts[selector].to('cpu'))
    vis = visualizer.draw_instance_predictions(insts.to('cpu'))
    os.makedirs('images', exist_ok=True)
    print(file_name)
    cv2.imwrite(f'images/gen_mask_prediction_{file_name}.png',
                vis.get_image()[:, :, ::-1])
    if fish:
        try:
            eyes = insts[insts.pred_classes == 2]
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
            results['fish_count'] = len(insts[(insts.pred_classes == 0).
                                        logical_and(insts.scores > 0.3)])

            # try:
            bbox = [round(x) for x in curr_fish.pred_boxes.tensor.cpu().
                numpy().astype('float64')[0]]
            im_crop = im_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            detectron_mask = curr_fish.pred_masks[0].cpu().numpy()
            val = adaptive_threshold(bbox, im_gray)
            bbox, mask, pixel_anal_failed = gen_mask(bbox, file_path,
                                                     file_name, im_gray, val, detectron_mask, index=i)
            # except:
            # return {file_name: {'errored': True}}
            if not np.count_nonzero(mask):
                print('Mask failed: {file_name}')
                results['errored'] = True
            else:
                # print(mask)
                im_crop = im_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]].reshape(-1)
                mask_crop = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].reshape(-1)
                # print(list(zip(list(im_crop),list(mask_crop))))
                # print(np.count_nonzero(mask_crop))
                fground = im_crop[np.where(mask_crop)]
                bground = im_crop[np.where(np.logical_not(mask_crop))]
                # print(im_crop.shape)
                # print(fground.shape)
                # print(bground.shape)
                results['fish'][i]['foreground'] = {}
                results['fish'][i]['foreground']['mean'] = np.mean(fground)
                results['fish'][i]['foreground']['std'] = np.std(fground)
                results['fish'][i]['background'] = {}
                results['fish'][i]['background']['mean'] = np.mean(bground)
                results['fish'][i]['background']['std'] = np.std(bground)
                results['fish'][i]['bbox'] = list(bbox)
                results['fish'][i]['pixel_analysis_failed'] = pixel_anal_failed
                start, code = encoded_mask(mask)
                results['fish'][i]['mask'] = {}
                results['fish'][i]['mask']['start_coord'] = list(start)
                results['fish'][i]['mask']['encoding'] = code
                # results['fish'][i]['mask'] = '[...]'

                centroid, evecs, length, width, area = pca(mask, scale)
                major, minor = evecs[0], evecs[1]
                if scale:
                    results['fish'][i]['length'] = length
                    results['fish'][i]['width'] = width
                    results['fish'][i]['area'] = area
                    results['fish'][i]['perimeter'] = perimeter(code, scale)
                    results['fish'][i]['bbox_length'] = fish_box_length(mask, centroid, major, scale)
                    results['fish'][i]['bbox_width'] = fish_box_length(mask, centroid, minor, scale)

                results['fish'][i]['centroid'] = centroid.tolist()
                if eye:
                    # print(eye.pred_boxes.get_centers())
                    eye_center = [round(x) for x in
                                  eye.pred_boxes.get_centers()[0].cpu().numpy()]
                    results['fish'][i]['eye_center'] = list(eye_center)
                    dist1 = distance(centroid, eye_center + major)
                    dist2 = distance(centroid, eye_center - major)
                    if dist2 > dist1:
                        # print("HERE")
                        # print(evec)
                        major *= -1
                        # print(evec)
                    if major[0] <= 0.0:
                        results['fish'][i]['side'] = 'left'
                    else:
                        results['fish'][i]['side'] = 'right'
                    x_mid = int(bbox[0] + (bbox[2] - bbox[0]) / 2)
                    y_mid = int(bbox[1] + (bbox[3] - bbox[1]) / 2)
                    # snout_vec = find_snout_vec(np.array([x_mid, y_mid]), eye_center, mask)
                    snout_vec = major
                    if snout_vec is None:
                        results['fish'][i]['clock_value'] = \
                            clock_value(major, file_name)
                    else:
                        results['fish'][i]['clock_value'] = \
                            clock_value(snout_vec, file_name)
                results['fish'][i]['primary_axis'] = list(major)
                # print(curr_fish)
                results['fish'][i]['score'] = float(curr_fish.scores[0].cpu())
                # print(results['fish'][i]['score'])
    # pprint.pprint(results)
    return {file_name: results}


def adaptive_threshold(bbox, im_gray):
    """
    Determines the best thresholding value.
    Parameters:
        bbox -- bounding box in [top left x, top left y, bottom right x, bottom right y] format.
        im_gray -- grayscale version of original image.
    Returns:
        val -- new threshold.
    """
    # bbox_d = [round(x) for x in curr_fish.pred_boxes.tensor.cpu().
    # numpy().astype('float64')[0]]
    im_crop = im_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    val = filters.threshold_otsu(im_crop)
    mask = np.where(im_crop > val, 1, 0).astype(np.uint8)
    # f_bbox_crop = curr_fish.pred_masks[0].cpu().numpy()\
    # [bbox_d[1]:bbox_d[3],bbox_d[0]:bbox_d[2]]
    flat_mask = mask.reshape(-1)
    # fground = im_crop.reshape(-1)[np.where(flat_mask)]
    bground = im_crop.reshape(-1)[np.where(np.logical_not(flat_mask))]
    mean_b = np.mean(bground)
    # mean_f = np.mean(fground)
    # print(f'b: {mean_b} | f: {mean_f}')
    # flipped = mean_b < mean_f
    flipped = False
    diff = abs(mean_b - val)
    # print(diff)
    # val = (mean_b + mean_f) / 2
    if flipped:
        val -= diff * VAL_SCALE_FAC
    else:
        val += diff * VAL_SCALE_FAC
    val = min(max(1, val), 254)
    return val


def find_snout_vec(centroid, eye_center, mask):
    """
    Determine the direction of the snout.
    Parameters:
        centroid -- center of fish in [x, y] format.
        eye_center -- center of eye in [x, y] format.
        mask -- thresholded image.
    Returns:
        max_vec / max_len -- vector pointing in direction of snout.
    """
    eye_dir = eye_center - centroid
    x1 = centroid[0]
    y1 = centroid[1]
    # print(centroid)
    # print(eye_center)
    # print(eye_dir)
    max_len = 0
    # fallback = np.array([-1,0])
    max_vec = None
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            # print((x, y))
            if mask[y, x]:
                x2 = x
                y2 = y
                curr_dir = np.array([x2 - x1, y2 - y1])
                curr_eye_dir = np.array([x2 - eye_center[0],
                                         y2 - eye_center[1]])
                curr_len = np.linalg.norm(curr_dir)
                if curr_len > max_len:
                    fallback = curr_dir
                    max_len = curr_len
                    if curr_len > np.linalg.norm(curr_eye_dir):
                        max_vec = curr_dir
    # print(max_vec)
    if max_len == 0:
        # return np.array([-1,0])
        return None
    if max_vec is None:
        print(f'Failed snout')
        # max_vec = fallback
        return None
    return max_vec / max_len


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
    # print(evec)
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
    # print(comp)
    ang = angle(comp, evec)
    # print(ang / (2 * math.pi) * 12)
    clock = start + (ang / (2 * math.pi) * 12)
    # print(clock)
    if clock > 11.5:
        clock = 12
    elif clock < 0.5:
        clock = 12
    # print(evec)
    return round(clock)


def fish_box_length(mask, centroid, evec, scale):
    """
    Check how far fish pixels gets in each direction from the centroid of
    the fish blob then return fish length. This is done by
    intersection the major axis with a line defined by a given fish pixel
    and the minor axis, then finding which two intersection points are
    farthest from the centroid in each direction.
    Parameters:
        mask -- thresholded image.
        centroid -- center of fish in [x, y] format.
        evec -- major axis of fish.
        scale -- pixels per unit.
    Returns:
        distance -- distance from max to min points on major axis.
    """
    m1 = evec[1] / evec[0]
    m2 = evec[0] / evec[1]
    # Set these as the first point for point slope form of a line
    # to be used with m1
    x1 = centroid[0]
    y1 = centroid[1]
    # Initial values for how far from the major axis
    # points project in each direction
    x_min = centroid[0]
    x_max = centroid[0]
    # Loop over every pixel in the bounding box
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            # If it is a fish pixel
            if mask[y, x]:
                # Set this as the second point for point slope form of a line
                # to be sued with m2
                x2 = x
                y2 = y
                # Intersect the major axis with the line formed by x2, y2 and
                # m2. I calculated this using basic algebra given the two
                # line equations.
                x_calc = (-y1 + y2 + m1 * x1 - m2 * x2) / (m1 - m2)
                y_calc = m1 * (x_calc - x1) + y1
                # If this is the new furthest point in one or the other,
                # save it
                if x_calc > x_max:
                    x_max = x_calc
                    y_max = y_calc
                elif x_calc < x_min:
                    x_min = x_calc
                    y_min = y_calc
    # Return the distance between the points we've found scaled into cms
    return distance((x_max, y_max), (x_min, y_min)) / scale


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
    # print(np.count_nonzero(img))
    moments = cv2.moments(img)
    centroid = (int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"]))
    # print(centroid)
    y, x = np.nonzero(img)

    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])

    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue

    # theta = np.arctan(x_v1 / y_v1)
    theta = np.arctan2(y_v1, x_v1)
    # print(x_v1, y_v1, theta, np.linalg.norm(evecs[:, sort_indices[0]]))
    # negate for clockwise rotation
    if y_v1 * theta > 0:
        theta *= -1
    rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
    transformed_mat = rotation_mat * coords
    # plot the transformed blob
    x_transformed, y_transformed = transformed_mat.A
    x_round, y_round = x_transformed.round(decimals=0), y_transformed.round(decimals=0)
    x_vals, x_counts = np.unique(x_round, return_counts=True)
    y_vals, y_counts = np.unique(y_round, return_counts=True)
    x_calc, y_calc = x_vals[x_counts.argmax()], y_vals[y_counts.argmax()]
    x_indices, y_indices = np.where(x_round == x_calc), np.where(y_round == y_calc)
    width = y_round[x_indices].max() - y_round[x_indices].min()
    length = x_round[y_indices].max() - x_round[y_indices].min()

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
        length /= glob_scale
        width /= glob_scale
        area /= glob_scale ** 2

    return np.array(centroid), evecs[:, sort_indices], length, width, area


def find_nearest(array, value):
    '''
    Find the nearest element of array to the given value
    '''
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def encode_freeman(image_contour):
    '''
    Encode the image contour in an 8-direction freeman chain code based on angles

    '''
    freeman_code = ""
    freeman_dict = {-90: '0', -45: '1', 0: '2', 45: '3', 90: '4', 135: '5', 180: '6', -135: '7'}
    allowed_directions = np.array([0, 45, 90, 135, 180, -45, -90, -135])

    for i in range(len(image_contour) - 1):
        delta_x = image_contour[i + 1][1] - image_contour[i][1]
        delta_y = image_contour[i + 1][0] - image_contour[i][0]
        angle = allowed_directions[np.abs(allowed_directions - np.rad2deg(np.arctan2(delta_y, delta_x))).argmin()]
        if not (delta_x == 0 and delta_y == 0):
            freeman_code += freeman_dict[angle]

    return freeman_code


def encoded_mask(mask, visualize=False):
    # Extract the longest contour in the image
    contours = measure.find_contours(mask, 0.9)
    contours_main = np.around(max(contours, key=len), decimals=0)

    if visualize:
        # Display the image and plot the main contour found
        fig, ax = plt.subplots()
        ax.imshow(mask, cmap=plt.cm.gray)
        ax.plot(contours_main[:, 1], contours_main[:, 0])

    # Extract freeman code from contour
    return contours_main[0][::-1], encode_freeman(contours_main)


def decode_freeman(contours, mask, code, visualize=False):
    coords = [list(contours[0][::-1])]
    freeman_dict = {0: [0, -1], 1: [1, -1], 2: [1, 0], 3: [1, 1], 4: [0, 1], 5: [-1, 1], 6: [-1, 0], 7: [-1, -1]}
    for letter in code:
        change = freeman_dict[int(letter)]
        current = coords[-1]
        coords.append([current[0] + change[0], current[1] + change[1]])
    coords = np.array(coords)
    if visualize:
        fig, ax = plt.subplots()
        ax.imshow(mask, cmap=plt.cm.gray)
        ax.plot(coords[:, 0], coords[:, 1])
        plt.show()
    return np.array(coords)


def perimeter(code, scale):
    even_numbers = ''.join(filter(lambda x: int(x) % 2 == 0 and int(x) > 0, code.split()))
    odd_numbers = ''.join(filter(lambda x: int(x) % 2 == 1 and int(x) > 0, code.split()))
    return (len(even_numbers) + np.sqrt(2) * len(odd_numbers)) / scale


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
        # print("Converting in to cm")
    elif any(name in file_name for name in cm_list):
        pass
        # print("Already in cm")
    else:
        scale /= 2.54
        print("Unable to determine unit. Defaulting to cm.")
    # print(f'Pixels/cm: {scale}')
    return scale


def check(arr, val, flipped):
    if flipped:
        return arr > val
    return arr < val


def gen_mask(bbox, file_path, file_name, im_gray, val, detectron_mask,
             index=0, flipped=False):
    """
    Generates the mask for the fish and floodfills to make a whole image.
    """
    failed = False
    l = round(bbox[0])
    r = round(bbox[2])
    t = round(bbox[1])
    b = round(bbox[3])
    bbox_orig = bbox
    bbox = (l, t, r, b)

    im = Image.open(file_path).convert('L')
    arr2 = np.array(im)
    shape = arr2.shape
    done = False
    im_crop = im_gray[t:b, l:r]
    fish_pix = None
    while not done:
        done = True
        arr0 = np.array(im.crop(bbox))
        bb_size = arr0.size

        arr1 = np.where(arr0 < val, 1, 0).astype(np.uint8)
        indices = list(zip(*np.where(arr1 == 1)))
        shuffle(indices)
        count = 0
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
            temp = flood_fill(arr1, ind, 2)
            temp = np.where(temp == 2, 1, 0)
            percent = np.count_nonzero(temp) / bb_size
            if percent > 0.1:
                fish_pix = ind
                # Flood fills from each of the bbox corners
                for i in (0, temp.shape[0] - 1):
                    for j in (0, temp.shape[1] - 1):
                        temp = flood_fill(temp, (i, j), 2)

                arr1 = np.where(temp != 2, 1, 0).astype(np.uint8)
                break
        arr3 = np.full(shape, 0).astype(np.uint8)
        arr3[t:b, l:r] = arr1

        # Expands the bounding box
        try:
            if np.any(arr3[t:b, l] != 0) and l > 0:
                l -= 1
                l = max(0, l)
                done = False
            if np.any(arr3[t:b, r] != 0) and r < shape[1] - 1:
                r += 1
                r = min(shape[1] - 1, r)
                done = False
            if np.any(arr3[t, l:r] != 0) and t > 0:
                t -= 1
                t = max(0, t)
                done = False
            if np.any(arr3[b, l:r] != 0) and b < shape[0] - 1:
                b += 1
                b = min(shape[0] - 1, b)
                done = False
        except:
            print(f'{file_name}: Error expanding bounding box')
            # done = True
            return bbox_orig, detectron_mask.astype('uint8'), True
        # New bbox
        bbox = (l, t, r, b)
        # New threshold
        val = adaptive_threshold(bbox, im_gray)
    if np.count_nonzero(arr1) / bb_size < .1:
        print(f'{file_name}: Using detectron mask and bbox')
        arr3 = detectron_mask.astype('uint8')
        bbox = bbox_orig
        failed = True
    arr4 = np.where(arr3 == 1, 255, 0).astype(np.uint8)
    (l, t, r, b) = shrink_bbox(arr3)
    arr4[t:b, l] = 175
    arr4[t:b, r] = 175
    arr4[t, l:r] = 175
    arr4[b, l:r] = 175
    im2 = Image.fromarray(arr4, 'L')
    im2.save(f'images/gen_mask_mask_{file_name}_{index}.png')
    return bbox, arr3, failed


# https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def shrink_bbox(mask):
    """
    Finds the bounding box of an image.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return cmin, rmin, cmax, rmax


def gen_metadata_safe(file_path):
    """
    Deals with erroneous metadata generation errors.
    """
    try:
        return gen_metadata(file_path)
    except Exception as e:
        print(f'{file_path}: Errored out ({e})')
        return {file_path: {'errored': True}}


def main():
    direct = sys.argv[1]
    if os.path.isdir(direct):
        files = [entry.path for entry in os.scandir(direct)]
        if len(sys.argv) > 2:
            files = files[:int(sys.argv[2])]
    else:
        files = [direct]
    # print(files)
    # predictor = init_model()
    # f = partial(gen_metadata, predictor)
    with Pool(3) as p:
        # results = map(gen_metadata, files)
        results = p.map(gen_metadata_safe, files)
    # results = map(gen_metadata, files)
    output = {}
    for i in results:
        output[list(i.keys())[0]] = list(i.values())[0]
    # print(output)
    if len(output) > 1:
        with open('metadata.json', 'w') as f:
            json.dump(output, f)
    else:
        pprint.pprint(output)


if __name__ == '__main__':
    # gen_metadata(sys.argv[1])
    main()
