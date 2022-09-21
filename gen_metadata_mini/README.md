# Refactor version of the original code

This specific folder contianed a simplified version of the [original code](https://github.com/hdr-bgnn/drexel_metadata), that suits better the needs of the main BGNN project "the Minnows project".
Additionally we have refactored the code to improve readability and flexibility. Finally two papers have been published related to the original work.

+ [Joel Pepper et .al](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9651834&casa_token=csQdLvi9K2cAAAAA:pMWSDHozeLugmFMNiMTChZiVf-6OjSl8zoouLyV_DRsPp_q3GvNL0Fvloe___GbaMRSiPA&tag=1)
+ [Kevin Karnani et  .al](https://assets.researchsquare.com/files/rs-1506561/v1_covered.pdf?c=1651071974)


# 1- Introduction

Using detectron2 framework (an object detection tool), the original auhtors have trained a model to detect fish and ruler from a fish image obtained from museum dataset, the image used for training test and prediction are available on Tulane server. Here an example. An example of model prediction is here. 
![prediction image](https://github.com/thibaulttabarin/drexel_metadata/blob/main/gen_metadata_mini/image_test/prediction_50577.jpg)

- The training phase was developped by Drexel group (Joel Pepper and Kevin Karnina). They used original images from tulane server and annotations were constructed using makesense.ai interface which produce coco.json format. More detail of how the training is performed is described in the following [tutorial](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) and [colab](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5).
    
- The prediction step predicts 5 classes {fish, eye, ruler, "number 2, "number 3"} in the form of instances. each instances is accessible individually and features properties such as bounding box mask center of bounding box... 
    
- The next step using a pixel analysis to refine the contour of the fish. [mask_example](https://github.com/thibaulttabarin/drexel_metadata/blob/main/gen_metadata_mini/image_test/mask_50577.png). Keep in mind that the pixel analysis sometime fail.
- Last step is concerned with collecting the metadata and calculating some interesting measurement using pixel analysis mask, object instance properties combine with classic computer vision approach. For instance we extract bounding box around the fish, eye center, fish orientation, scale of the ruler, background average pixel value... More exhautive list of possible output is described on the [original repo](https://github.com/hdr-bgnn/drexel_metadata)

- The metadata output are in the format of json file (equivalent to dictionary format in python) [metadta example](https://github.com/thibaulttabarin/drexel_metadata/blob/main/gen_metadata_mini/image_test/metadata_50577.json)

# 2- About this version

This version of the code reuses [original code](https://github.com/hdr-bgnn/drexel_metadata) developped by Joel and Kevin [here](https://github.com/hdr-bgnn/drexel_metadata). Here, we reuse the model weights for the prediction part. We have modified some part of the code to improve readability and simplify some functionalities to output subset of the output more relevant for the [BGNN_Snakemake project](https://github.com/hdr-bgnn/BGNN_Snakemake) and [Minnows project](https://github.com/hdr-bgnn/Minnow_Traits). For the BGNN_Snakemake and Minnows project we focus on extracting form the image the fish bounding box, the orientation, the eye orientation and scale (of the ruler). Noticeable modification
+ The simplify output
    - Json file metadata contained : {base_name: , fish:{}, ruler:{}} [example here](https://github.com/thibaulttabarin/drexel_metadata/blob/main/gen_metadata_mini/image_test/metadata_50577.json)
    - PNG file for the mask of the file : [example here](https://github.com/thibaulttabarin/drexel_metadata/blob/main/gen_metadata_mini/image_test/mask_50577.png)
+ Break the code in smaller functions to add modularity.
+ Remove part that weren't use in our application for the [Minnows project](https://github.com/hdr-bgnn/Minnow_Traits) 


# 3- Model weights: 

In this repository folder we are using the follwing model. (Other model have been used, check the original repo and with the author for more info on the other model).  Our model of interest was trained by Kevin Karnina using INHS and UWZM dataset and using a contract enhancement fucntion.

More information on the training are available on the [original repo](https://github.com/hdr-bgnn/drexel_metadata)

Location of the model is on ohio state university https://datacommons.tdai.osu.edu/dataset.xhtml?persistentId=doi:10.5072/FK2/MMX6FY&version=DRAFT. Currently, the model is unpublished, therefore to acccess the weights you need account (contact Hilmar , John or Thibault). The model should be published in a near future. However you can use the container to run the code (generate the metadata). See next section
+ Instruction to get the weights from google drive (current public location:
   - Require pip to install gdown
   - Run load_models.sh to get the weights
+ Instruction to download the weights from OSC using pydataverse python module
    + use the functions developped [here](https://github.com/thibaulttabarin/drexel_metadata/blob/main/gen_metadata_mini/scripts/dataverse_download.py), [here](https://github.com/johnbradley/BGNN-trait-segmentation/commit/c6aa67663694557136e0573cff2b0072d5645143) or [read documentation](https://pydataverse.readthedocs.io/en/latest/)
    + you will need API_TOKEN [instruction](https://guides.dataverse.org/en/latest/api/auth.html) 

# 3- Suggested Setup and Requirements with anaconda 

The set up for detectron2 can be a bit difficult. At the time of this project the best option found was to use conda and manually install the dependencies. The [original repo](https://github.com/hdr-bgnn/drexel_metadata/blob/main/Pipfile) was using the pipfile system. It is up to you! 

   - Require conda or miniconda
   - Run env_setup.sh to setup an environment named gen_metadata
   - Or execute each command contained in env_setup.sh
   
# 3 Usage and output: 

Activate your environment  
```
conda activate gen_metadata
python  metadata_main.py INHS_FISH_50577.jpg result_metadata.json mask.png
```

This will generate 2 files:

    - result_metadata.json : contained various metadata information. fish bounding box, scale bounding box, scale conversion (pixel/cm)
    - mask.png : improve fish mask using the pixel analysis. (binary map)
    
[Metadata](https://github.com/thibaulttabarin/drexel_metadata/blob/main/gen_metadata_mini/image_test/metadata_50577.json)

## Properties Generated
| Key                   | Association   | Type    | Explanation                                                            |
|:----------------------|:--------------|:--------|:-----------------------------------------------------------------------|
| Base_name             | Overal image  | string  | image name without extension                                           |
| fish                  | Fish          | dict    | collect metadata of the "main fish" with the highest score             |
| fish.fish_num         | Fish          | int     | number of fish detected in the image                                   |
| fish.bbox             | Fish          | list    | Bounding box of the main fish [left,top,right,bottom]                  |
| fish.pixel_analysis   | Fish          | boolean | If pixel analysis succeeded True, else False                           |
| fish.eye_bbox         | eye           | list    | Bounding box of the eye in the main fish [l,t,r,b]                     |
| fish.angle_degree     | Fish          | float   | angle of the PCA of the mask                                           |
| fish.eye_direction    | Fish          | string  | eye facing left or righ                                                |
| fish. foreground_mean | Fish          | float   | Average of pixel value inside the mask                                 |
| fish.foreground_std   | Fish          | float   | Standart deviation of pixel value inside the mask                      |
| ruler                 | Ruler         | dict    | collect metadata of the ruler                                          |
| ruler.bbox            | Ruler         | list    | Bounding box of the ruler [left,top,right,bottom]                      |
| ruler.scale           | Ruler         | float   | pixel/unit (distance between number "2" and "3" corrected by the unit) |
| ruler.unit            | Ruler         | string  | indicate unit (cm or inch) in which the scale is express (pixel/cm)    |

                                                                          

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
