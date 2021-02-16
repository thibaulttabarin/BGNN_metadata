import numpy as np
import cv2
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.data import Metadata
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
import os
from detectron2.utils.visualizer import ColorMode

PREFIX_DIR = '/home/HDD/bgnn_data/'
IMAGES_DIR = 'full_imgs/'
IMS = PREFIX_DIR + IMAGES_DIR

def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

"""
register_coco_instances('eyes', {}, PREFIX_DIR + 'eyes.json',
        IMS)
#for i in [1,2,3,4,5,6,7,8,9,10]:
for i in range(1,11):
    register_coco_instances(f'ruler{i}', {}, f'./data/{i}.json',
            IMS)
Metadata(evaluator_type='coco', image_root=IMS, json_file=PREFIX_DIR +
        'eyes.json', name='eyes', thing_classes=['fish', 'ruler', 'eye'],
        thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2})
"""
"""
Metadata(evaluator_type='coco', image_root='./data/images', json_file='./data/1.json', name='ruler1',
         thing_classes=['fish', 'ruler'], thing_dataset_id_to_contiguous_id={1: 0, 2: 1})
"""
"""
register_coco_instances('one_fish', {}, '/home/joel/School/Research/projects/bgnn/one_fish.json',
        IMS)
Metadata(evaluator_type='coco', image_root=IMS, json_file=
        '/home/joel/School/Research/projects/bgnn/one_fish.json', name='one_fish',
        thing_classes=['fish', 'ruler', 'eye'],
        thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2})
"""
register_coco_instances('two_three', {}, PREFIX_DIR + 'two_three.json',
        IMS)
Metadata(evaluator_type='coco', image_root=IMS, json_file=PREFIX_DIR + 'two_three.json', name='two_three',
         thing_classes=['two', 'three'], thing_dataset_id_to_contiguous_id={1: 0, 2: 1})

the_metadata = MetadataCatalog.get("two_three")
dataset_dicts = DatasetCatalog.get("two_three")

for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    print(d)
    visualizer = Visualizer(img[:, :, ::-1], metadata=the_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite('image.jpg', vis.get_image()[:, :, ::-1])

cfg = get_cfg()
cfg.merge_from_file(
    "/home/joel/detectron2_clean/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.DATASETS.TRAIN = ('two_three',)
#cfg.DATASETS.TRAIN = tuple([f"ruler{i}" for i in range(1,11)] + ['eyes'])
#cfg.DATASETS.TRAIN = tuple([f"ruler{i}" for i in [1,2,3,4,5,6,10]] + ['eyes'])
#cfg.DATASETS.TRAIN = ('eyes',)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = (
    400
)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)  # faster, and good enough for this toy dataset
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

i = 0
#loc = '/home/HDD/bgnn_data/other_museums/osum/osum1/'
loc = '/home/HDD/bgnn_data/full_imgs_grouped/4/'
names = os.listdir(loc)
#names = [i.split('.')[0] for i in segments]

outputs = None
for d in random.sample(names, 10):
#for d in ['INHS_FISH_008363.jpg', 'INHS_FISH_001980.jpg']:
#for d in ['INHS_FISH_000452.jpg']:
    #d = '/home/HDD/bgnn_data/full_imgs_grouped/bad/head.jpg'
    #im = cv2.imread(PREFIX_DIR + 'full_imgs_large/' + d)
    im = cv2.imread(IMS + d)
    outputs = predictor(im)
    #print(outputs)
    v = Visualizer(im[:, :, ::-1],
                   metadata=the_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    i+=1
    print(f'{i}: {d}')
    pt1 = outputs['instances'][0].get('pred_boxes').get_centers()
    pt2 = outputs['instances'][1].get('pred_boxes').get_centers()
    print(f'\tPixels/Inch: {distance([float(pt1[0][0]), float(pt1[0][1])], [float(pt2[0][0]), float(pt2[0][1])])}')
    cv2.imwrite(f'temp/testing2/{d}', v.get_image()[:, :, ::-1])
    #exit(0)

"""
for d in random.sample(dataset_dicts, 200):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    print(outputs)
    if len(outputs['instances']):
        v = Visualizer(im[:, :, ::-1],
                       metadata=the_metadata,
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        print("HERE")
        x = f'temp/pred{i}.jpg'
        i+=1
        cv2.imwrite(x, v.get_image()[:, :, ::-1])
        if i > 3:
            exit(0)
"""
