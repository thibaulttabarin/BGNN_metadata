# User Branch for Gen_metadata
This specific folder contained the minimun necessary to start using the generator of metadata for fish

# 1- Introduction

Using detectron2 framework (an object detection tool), the original auhtors have trained a model to detect fish and ruler from a fish image obtained from museum dataset, the image used for training test and prediction are available on Tulane server. Here an example. An example of model prediction is here. 


- The training phase was developped by Drexel group (Joel Pepper and Kevin Karnina). They used original images from tulane server and annotations were constructed using makesense.ai interface which produce coco.json format. More detail of how the training is performed is described in the following [tutorial](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) and [colab](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5).
    
- The prediction step predicts 5 classes {fish, eye, ruler, "number 2, "number 3"} in the form of instances. each instances is accessible individually and features properties such as bounding box mask center of bounding box... 
    
- The next step is the metadata extraction. Using object instance properties and classic computer vision approach, we extract bounding box around the fish, eye center, fish orientation, scale of the ruler, background average pixel value... More exhautive list of possible output is described on the [original repo](https://github.com/hdr-bgnn/drexel_metadata)

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

![mask](https://github.com/thibaulttabarin/drexel_metadata/blob/main/gen_metadata_mini/image_test/mask_50577.png)
![Metadata](https://github.com/thibaulttabarin/drexel_metadata/blob/main/gen_metadata_mini/image_test/metadata_50577.json)
 
# 4- Container

The code has been contianerized and the image is available [here](https://github.com/thibaulttabarin/drexel_metadata/pkgs/container/drexel_metadata)

Pull command
```
docker pull ghcr.io/thibaulttabarin/drexel_metadata:main
singularity pull docker://ghcr.io/thibaulttabarin/drexel_metadata:main
```

Container Usage (for singularity:
```
singularity exec drexel_metadata_main.sif metadata_main.py INHS_FISH_50577.jpg result_metadata.json mask.png
``` 
 

