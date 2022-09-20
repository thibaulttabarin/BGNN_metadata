# User Branch for Gen_metadata
This specific folder contained the minimun necessary to start using the generator of metadata for fish

# 1- Introduction

Using detectron2 framework (an object detection tool), the original auhtors have trained a model to detect fish and ruler from a fish image obtained from museum dataset, the image used for training test and prediction are available on Tulane server. Here an example. An example of model prediction is here. 
![prediction image](https://github.com/thibaulttabarin/drexel_metadata/blob/main/gen_metadata_mini/image_test/prediction_50577.jpg)

- The training phase was developped by Drexel group (Joel Pepper and Kevin Karnina). They used original images from tulane server and annotations were constructed using makesense.ai interface which produce coco.json format. More detail of how the training is performed is described in the following [tutorial](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) and [colab](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5).
    
- The prediction step predicts 5 classes {fish, eye, ruler, "number 2, "number 3"} in the form of instances. each instances is accessible individually and features properties such as bounding box mask center of bounding box... 
    
- The next step using a pixel analysis to refine the contour of the fish. [mask_example](https://github.com/thibaulttabarin/drexel_metadata/blob/main/gen_metadata_mini/image_test/mask_50577.png). Keep in mind that the pixel analysis sometime fail.
- Last step is concerned with collecting the metadata and calculating some interesting measurement using pixel analysis mask, object instance properties combine with classic computer vision approach. For instance we extract bounding box around the fish, eye center, fish orientation, scale of the ruler, background average pixel value... More exhautive list of possible output is described on the [original repo](https://github.com/hdr-bgnn/drexel_metadata)

- The metadata output are in the format of json file (equivalent to dictionary format in python)

# 2- About this version

This version of the code reuses original code developped by Joel and Kevin [here](https://github.com/hdr-bgnn/drexel_metadata). Here, we ruse the model weights for the prediction part. We have modified some part of the code to improve readability and simplify some functionalities to output subset of the output more relevant for the [BGNN_Snakemake project](https://github.com/hdr-bgnn/BGNN_Snakemake). For the BGNN_Snakemake and Minnows project we focus on extracting form the image the fish bounding box, the orientation, the eye orientation and scale (of the ruler). Noticeable modification
+ The simplify output
    - Json file metadata contained : {base_name: , fish:{}, ruler:{}} [example here]()
    - PNG file for the mask of the file : [example here]()
+ Break the code in smaller functions
+ Remove part that weren't use in our application for 


# 3- Model weights: 

In this repository folder we are using the follwing model. (other model have been used check the original repo and with the author for more info on the other model.  Our model of interest was trained by Kevin Karnina using INHS and UWZM dataset and using a contract enhancement fucntion.

More information on the training are available on the [original repo](https://github.com/hdr-bgnn/drexel_metadata)

Location of the model is on ohio state university https://datacommons.tdai.osu.edu/dataset.xhtml?persistentId=doi:10.5072/FK2/MMX6FY&version=DRAFT. Current the model is unpublished, therefore to acccess the weights you need account (contact Hilmar , John or Thibault). The model should be published in a near future. However you can use the container to run the code. See next section
   - Require pip to install gdown
   - Run load_models.sh to get the weights
   
# 3- Setup and Requirements

   - Require conda or miniconda
   - Run env_setup.sh to setup an environment named gen_metadata
   - Or execute each command contained in env_setup.sh
   
# 3 Usage and output: 

Activate your environment  
```
conda activate gen_metadata
python  INHS_FISH_50577.jpg result_metadata.json mask.png
```

This will generate 2 files:

    - result_metadata.json : contained various metadata information. fish bounding box, scale bounding box, scale conversion (pixel/cm)
    - mask.png : improve fish mask using the pixel analysis. (binary map)
    
[Metadata](https://github.com/thibaulttabarin/drexel_metadata/blob/main/gen_metadata_mini/image_test/metadata_50577.json)
{"base_name": "INHS_FISH_50577", "fish": {"fish_num": 1, "bbox": [1031, 303, 4652, 1696], "pixel_analysis": true, "eye_bbox": [1227, 713, 1572, 1041], "eye_center": [1399, 877], "angle_degree": -1.59, "eye_direction": "left", "foreground_mean": 105.12, "foreground_std": 45.23, "background_mean": 242.33, "background_std": 13.96}, "ruler": {"bbox": [319, 2601, 3446, 3664], "scale": 339.88, "unit": "cm converted"}}

## Properties Generated

| **Property**            | **Association** | **Type** | **Explanation**                                                                                                                                   |
|----------------------------------|--------------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| base\_name              | Overall Image            | string           | image name.                                                                                                                     |
| fish                    | -------------            | ------           | ------------------                                                                                                                      |
| fish\_num               | Overall Image            | Integer           | The number of fish present.                                                                                                                      |
| bbox                    | fish of interest         | list              | fish bounding box (top, left, right, bottom). 
                                                                             |
| pixel_analysis          | fish of interest         | Booleen           | If pixel analysis succeeded True, else False.
                                                                             |                                         
| eye_bbox                | for eye in fish          | list              | bounding box around the eye with best overlap with the fish.                                                                                                |                                                          
                                                                             
## Properties Generated

| **Property**            | **Association** | **Type** | **Explanation**                                                                                                                                   |
|----------------------------------|--------------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| base\_name              | Overall Image            | string           | image name.

| fish\_num               | Overall Image            | Integer           | The number of fish present.                                                                                                                               
| bbox                    | fish of interest         | list              | fish bounding box [top, left, right, bottom] highest "confidence" score.
|
| pixel_analysis          | fish of interest         | Booleen           | If pixel analysis succeeded True, else False
|
| eye_bbox                | for eye in fish          | list              | bounding box around the eye with best overlap with the fish.
|
| eye_center              | for eye in fish          | list              | Center of the eye.                                                              |                                                 
| angle_degree            | for fish                 | Float             | Angle orientation of the PCA of the mask (in degree).                |                                                                          
| eye_direction           | for Fish                 | string            |      |                                                                              
| background.mean         | Per Fish                 | Float             | The mean intensity of the background within a given fish's bounding box.       |                                                                             
| background.std          | Per Fish                 | Float             | The standard deviation of the background within a given fish's bounding box.    |                                                                            
| foreground.mean         | Per Fish                 | Float             | The mean intensity of the foreground within a given fish's bounding box.       |                                                                             
| foreground.std          | Per Fish                 | Float             | The scale of the image in $\frac{\mathrm{pixels}}{\mathrm{cm}}$The standard deviation of the foreground within a given fish's bounding box.                                                                                


![mask](https://github.com/thibaulttabarin/drexel_metadata/blob/main/gen_metadata_mini/image_test/mask_50577.png)

# 4- Container

The code has been containerized and the image is available [here](https://github.com/thibaulttabarin/drexel_metadata/pkgs/container/drexel_metadata). Check for more updated version. Here we are using "main" but new release may be available (tag looks like this "0.0.10")

Pull command
```
docker pull ghcr.io/thibaulttabarin/drexel_metadata:main
singularity pull docker://ghcr.io/thibaulttabarin/drexel_metadata:main
```

Container Usage (for singularity):
```
singularity exec drexel_metadata_main.sif metadata_main.py INHS_FISH_50577.jpg result_metadata.json mask.png
``` 
