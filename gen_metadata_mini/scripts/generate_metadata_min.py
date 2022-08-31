#!/usr/local/bin/python
#' Refactoredls on Mon Aug  8 11:35:50 2022
#' @author
#' Joel Pepper: initial code
#' Kevin Karnani: modified it
#' Thibault Tabarin: modularize it and trimmed the code for minnow project

#' @description
#' Using the detectron2 framework from Facebook, we detect fish, eye and ruler object and
#' collect metadata information such bounding box, fish mask, eye center, fish orientation...
#' The main outputs are 
#' Dictionnary with metadata:  {"base_name": "", "fish": 
#' {"fish_num": , "bbox": [], "pixel_analysis": true, "eye_bbox": [], "eye_center": [], 
#' "angle_degree": 9.070674226380035, "eye_direction": "left", "foreground_mean": ,
#' "foreground_std": , "background_mean": , "background_std": }, 
#' "ruler": {"bbox": [], "scale": , "unit": ""}}
#' And fish mask (binary image) 

import json
import math
import os
import sys
import torch
import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.data import Metadata
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import Boxes, pairwise_ioa
#from matplotlib import pyplot as plt # for development phase

import warnings
warnings.filterwarnings("ignore")
# setting path
try:
    root_file_path = os.path.dirname(__file__)
except:
    root_file_path = './'

sys.path.append(root_file_path)
import utility as ut

# import configuration
conf = json.load(open(os.path.join(root_file_path,'config/config.json'), 'r'))
PROCESSOR = conf['PROCESSOR']
MODEL_WEIGHT = os.path.join(root_file_path, conf['MODEL_WEIGHT'])
NUM_CLASSES = conf['NUM_CLASSES']
VAL_SCALE_FAC = conf['VAL_SCALE_FAC']


def init_model(processor=PROCESSOR, model_weight=MODEL_WEIGHT, NUM_CLASSES=5):
    '''
    Initialize model using config files for RCNN, the trained weights, and other parameters.

    Parameters
    ----------
    processor : string, optional
        type of processor to use; 'cpu', 'cuda'. The default is PROCESSOR.
    model_weight : string, optional
        Location of the model weight .pth. The default is MODEL_WEIGHT.
    NUM_CLASSES : int, optional
        Number of classes detected by the segmentation model. The default is 5.

    Returns
    -------
    predictor : detectron2.engine.defaults.DefaultPredictor
        Default predcitor use to detect object.

    '''
    #root_file_path = os.path.dirname(__file__)
    cfg = get_cfg()
    #cfg.merge_from_file(os.path.join(root_file_path,'config/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

    cfg.MODEL.WEIGHTS = model_weight
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.DEVICE = processor
    predictor = DefaultPredictor(cfg)

    return predictor


def create_metadata_obj():
    '''
        Create metadata object that contained info of the classes, and label numbers
    metadata object will be used for visualization

    Returns
    -------
    metadata : detectron2.data.catalog.Metadata
        detectron object that contains information on the data structure. More info on 
        detectron2 documentation.

    '''
    metadata = Metadata(evaluator_type='coco', image_root='.',
                    json_file='',
                    name='metadata',
                    thing_classes=['fish', 'ruler', 'eye', 'two', 'three'],
                    thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4})
    return metadata


def predict_detectron(file_path):
    '''
    import the image file, enhance imae contrast and predict the classes
    Classes : ['fish', 'ruler', 'eye', 'two', 'three']

    Parameters
    ----------
    file_path : string
        Loacation of the file (absolute path).

    Returns
    -------
    insts : dectetron object detectron2.structures.instances.Instances
        List of dict of list...
    im : np.ndarray (dtype=uint8)
        Image loaded by cv2.
    '''

    # Initialize the model
    predictor = init_model()
    # load the image
    im = cv2.imread(file_path)
    # CLAHE + prediction
    im_enh, im_gray = ut.enhance_contrast(im)
    output = predictor(im_enh)
    insts = output['instances']

    return insts, im


def create_prediction_image(im, insts):
    '''
    Create an image with the prediction boxes and scores

    Parameters
    ----------
    im : np.ndarray (dtype=uint8)
        Image loaded by cv2.
    insts : dectetron object detectron2.structures.instances.Instances
        List of dict of list...

    Returns
    -------
    pred_image : np.ndarray (dtype=uint8)
        Image with the prediction boxes and scores, check detectron2 documentation for
        more information.

    '''
    
    metadata = create_metadata_obj()
    v = Visualizer(im[:, :, ::-1], metadata=metadata,scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW)
    vis = v.draw_instance_predictions(insts.to('cpu'))
    pred_image = vis.get_image()[:, :, ::-1]

    return pred_image


def get_ruler_metadata(insts, file_name):
    '''
    Collect metatda related to the ruler

    Parameters
    ----------
    insts : detectron instances object
        Contained instances of object detected in each classes..
    file_name : str
        name of the file, this use to know scale unit depending of file origin.
    Returns
    -------
    dict_ruler : dictionnary
        Contain bbox of the ruler, scale value (distance in pixel between #2 and #3) and unit.

    '''
    dict_ruler={'bbox': "None", 'scale':"None", 'unit':"None"}

    # find the ruler
    try:
        ruler = insts[insts.pred_classes == 1][0]
        ruler_bbox = list(ruler.pred_boxes.tensor.cpu().numpy()[0])
        dict_ruler['bbox'] = [round(x) for x in ruler_bbox]
    except:
        ruler = None
    # find the number '2'
    try:
        two = insts[insts.pred_classes == 3][0]
    except:
        two = None
    # find the number '3'
    try:
        three = insts[insts.pred_classes == 4][0]
    except:
        three = None

    if ruler and two and three:
        scale, unit = ut.calculate_scale(two, three, file_name)
        dict_ruler['scale'] = scale
        dict_ruler['unit'] = unit

    return dict_ruler


def get_fish_metadata (insts, im):
    '''
    This function work for one fish
    Collect metadat from fish
    Choose the fish instance with high score from detectron and calculate/extract multiple metdata.
    number of fish, bbox of the main fish, mask_type (detecton/pixel_analysis), eye_bbox,
    fish orientation (angle and direction:left/right),

    Parameters
    ----------
    insts : detectron instances Structure
        Contained instances of object detected in each classes.
    im : np.ndarray (h,w,3)
        fish image in np.ndarray.

    Returns
    -------
    dict_fish : dictionnary
        {"fish_num" : 0, "bbox":"None", "mask_type":"None", "eye_bbox":"None",...}
    mask_uint8 : np.ndarray binary uint8
        mask of the fish.

    '''

    dict_fish={"fish_num" : 0, "bbox":"None", "pixel_analysis":"None",
               "eye_bbox":"None", "eye_center":"None"}

    # Select the fish with highest score
    main_fish_inst, num_fish = select_main_fish(insts)
    im_enhance, im_gray = ut.enhance_contrast(im)

    dict_fish['fish_num']=num_fish

    # if one or more fish, use main_fish_inst for pixel analysis
    if num_fish>=1:

        ### Fish
        # generate a new mask of the fish using pixel analysis
        mask_uint8, bbox, analysis_failed = generate_new_mask(main_fish_inst, im_gray)

        dict_fish ['bbox'] = bbox
        if analysis_failed:
            dict_fish ['pixel_analysis'] = False
        else:
            dict_fish ['pixel_analysis'] = True

        ### eye
        # Convert bbox list in Boxes structure
        Boxes_fish = Boxes(torch.tensor(bbox)[None,:])
        # find the main eye in the main fish
        main_eye_inst, num_eyes = find_main_eye(Boxes_fish, insts)
        # measure eye center and bbox
        eye_center = []
        if  main_eye_inst :
            eye_center = [round(x) for x in main_eye_inst.pred_boxes.get_centers()[0].cpu().numpy()]
            eye_bbox = [round(x) for x in main_eye_inst.pred_boxes.tensor.cpu().numpy()[0]]
            dict_fish ['eye_bbox'] = eye_bbox
            dict_fish ['eye_center'] = eye_center

        ## Orientation  {'angle_degree' : "None" , 'eye_direction' :"None" }
        dict_orientation = get_fish_orientation(mask_uint8, eye_center)
        # add the orientation metadata to the dict_fish
        dict_fish.update(dict_orientation)

        ## Brightness  {'foreground_mean':'None', 'foreground_std':'None',
        ##  'background_mean':'None', 'background_std':'None'}
        dict_brightness= ut.get_brightness(im, mask_uint8, bbox)
        dict_fish.update(dict_brightness)

    return dict_fish, mask_uint8


def select_main_fish(insts):
    '''
    Select the fish with highest score >0.3 from instance object generated by detectron

    Parameters
    ----------
    insts : detectron instance object (detectron2.structures.instances.Instances)
        Check detectron2 documentation, basically list of list of instance objects.

    Returns
    -------
    main_fish : detectron2.structures.instances.Instances
         .
    num_fish : int
        number of fish detected by detectron.

    '''
    fish = insts[insts.pred_classes == 0]
    num_fish = len(fish)
    main_fish=[]

    if num_fish>=1:
        # Select the fish with highest prediction score
        main_fish = fish[fish.scores.argmax().item()]
        # if fish instance scroe above 0.3 keep it
        if float(main_fish.scores)<0.3:
            main_fish=[]
            main_fish = 0

    return main_fish, num_fish

           
def generate_new_mask(fish, im_gray):
    '''
    Parameters
    ----------
    fish : instance from detectron
        This particular instance represents a fish (class =0) with highest score.
    im_gray : numpy array
        Gray image with contrast enhance from ut.enhance_contrast .

    Returns
    -------
    mask_uint8 : binary mask array with type uint8 (0, 255)
        mask of the fish output of ut.enhance_contrast.
    bbox : list
        new bounding box recaculate from the new mask_uint8.
    analysis_failed : BOOL
        True if the pixel analysis succeded.

    '''
    # convert the fish_instance mask into the numpy array
    detectron_mask = fish.pred_masks[0].cpu().numpy()
    # convert the fish_instance bbox into the list of float
    bbox = [round(x) for x in fish.pred_boxes.tensor.cpu().numpy().astype('float64')[0]]

    bbox, mask, analysis_failed = ut.generate_pixel_analysis(bbox, im_gray, VAL_SCALE_FAC, detectron_mask)
    mask_uint8 = np.where(mask == 1, 255, 0).astype(np.uint8)

    return mask_uint8, bbox, analysis_failed


def find_main_eye(Boxes_fish, insts):
    '''
    Find the eye with hightest score that overlap with  main_fish

    Parameters
    ----------
    fish_bbox : Boxes structure from detectron.structures
        single fish instance.
    insts : detectron instance object (detectron2.structures.instances.Instances)
        Check detectron2 documentation, basically list of list of instance objects.

    Returns
    -------
    main_eye : detectron instance (eye)
        single eye instance (that overlap with  main_fish).
    num_eyes : int
        number of eyes detected by detectron.

    '''

    main_eye = None

    eyes_insts = insts[insts.pred_classes == 2]
    num_eyes = len(eyes_insts)

    # eyes_insts are sort by score value in descending order
    for idx in range(len(eyes_insts)):


        # Boxes_fish and eye are Boxes structure from detectron2
        Boxes_eye = eyes_insts[idx].pred_boxes
        overlap_fish_eye = pairwise_ioa(Boxes_fish, Boxes_eye).item() # intersecton over area2 from detectron

        # find the first eye (highest score) in the fish (overlap>0.75)
        if overlap_fish_eye > 0.75:
            main_eye = eyes_insts[idx]
            break
    return main_eye, num_eyes


def get_fish_orientation(mask_fish, eye_center):
    '''
    Calculate the angle of the fish (using pca) regarding to horizontal line of the image and
    indicate the position of the eye right/left. The angle of the fish is reference to the horizontal
    line pointing to the left (standardized to the left because fishshould point to the left)
    Parameters
    ----------
    mask_fish : Binary mask array with type uint8 (0, 255)
        Mask of the fish output of ut.enhance_contrast.TYPE

    eye_center : list of int
        center of the eye.

    Returns
    -------
    dict_orientation : dictionnary
        {'angle_degree' : "None" , 'eye_direction' :"None" }.
        eye_direction is "None" if eye_center empty

    '''

    dict_orientation= {'angle_degree' : "None" , 'eye_direction' :"None" }

    # Clean the mask, convert to regionrops and select the biggest region
    biggest_region = ut.clean_regionprop(mask_fish)

    # Collect the orientation and convert to angle with horizontale facing left
    orientation = biggest_region.orientation
    angle = np.sign(orientation) * (90-abs(orientation*180/math.pi))
    dict_orientation["angle_degree"] = round(angle,2)

    ### get orientation left/right of the eye
    fish_center = biggest_region.centroid # [row,col]

    # eye_center format [col,row] is inverse compare to fish_center... don't ask!
    if eye_center and eye_center[0] < fish_center[1]:
        dict_orientation['eye_direction'] = 'left'
    elif  eye_center and eye_center[0] > fish_center[1]:
        dict_orientation['eye_direction'] = 'right'

    return dict_orientation


def main(file_path, output_json, output_mask):
    '''
    Combine the different step of the analysis
    1- import model and create a object detection prediction
    2- Collect metadata for the ruler
    3- Collect metadata for the fish and the fish mask
    4- Collect statistic from the fish mask
    5- Save the result in outputs (.json amd .png files)

    Parameters
    ----------
    file_path : string
        location of the imae file to analysis.
    output_json : string
        path for dictionnary output in json format (expected '/path/to/save/my_output.json').
    output_mask : string
        path for mask image output in png format (expected '/path/to/save/my_mask.png').

    Returns
    -------
    None.

    '''

    # try to run and save
    name_base = os.path.split(file_path)[1].rsplit('.',1)[0]
    result = {'base_name': name_base, 'status':'failed'}
    # empty mask if the analysis fail
    mask = np.zeros((100,100))
    try :
        insts, im = predict_detectron(file_path)
        # ruler metadata
        dict_ruler = get_ruler_metadata(insts, file_path)
        # fish
        dict_fish, mask = get_fish_metadata(insts, im)
        # Morphology and statistic
        #dict_morph_stat = ut.get_morphological_value(mask)

        result = {'base_name': name_base, 'fish': dict_fish, 'ruler': dict_ruler}

    except Exception as e:
        # write the error in the result dictionnary
        result['error'] = f'({e})'
                
    with open(output_json, 'w') as f:        
        json.dump(result, f)

    if output_mask != None:
        cv2.imwrite(output_mask, mask)


