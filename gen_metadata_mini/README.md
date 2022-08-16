# User Branch for Gen_metadata
This specific folder contained the minimun necessary to start using the generator of metadata for fish

# 1- Introduction

    More detail on kevin Branch

# 2- Setup and Requirements

   - Require conda or miniconda
   - Run the env_setup.sh
   - Or execute each command contained in env_setup.sh
   
# 3- Model weights: 

There are 3 models:  

    - One was developed by Joel with INHS dataset : output/final_model.pth
    - Two werer developed by Kevin using INHS and UWZM dataset. one model uses contrast enhancement: output/enhanced/final_model.pth 
    - The other doesn't use contrast enhancement : output/non_enhanced/final_model.pth

   - Require pip to install gdown
   - Run load_models.sh to get the weights

   
# 4 Usage: 

```
conda activate gen_metadata
python gen_metadata.py INHS_FISH_50577.jpg result_metadata.json mask.png
```

This will generate 2 files:

    - result_metadata.json : contained various metadata information. fish bounding box, scale bounding box, scale conversion (pixel/cm)
    - mask.png : improve fish mask using the pixel analysis. (binary map)
    - more detail of the metadata here https://github.com/hdr-bgnn/drexel_metadata/tree/kevin
 
# 5 Containers:

Several containers are available but you need singularity install on your system. In the repo we have provided the singularity recipes to create then:

    - detectron2_env.def : singularity recipe to create the base environment to run the code
        
        - Download the image from cloud.sylabs/thibaulttabairn/bgnn
        
```
singularity pull --arch amd64 library://thibaulttabarin/bgnn/detectron2_env:v1
```
        
        - Or recreate the image from detectron2_env.def 
        
```
sudo singularity build detectron2_env.sif detectron2_env.def
```

        
    - gen_metadata.def : package the code and the weight snecessary to run execute the gen_metadata.py
    
        - Download the image from cloud.sylabs/thibaulttabairn/bgnn
        
```
singularity pull --arch amd64 library://thibaulttabarin/bgnn/gen_metadata:v2
```
        
        - Or recreate the image from gen_metadata.def 
        
```
sudo singularity build --force gen_metadata_v2.sif gen_metadata.def
```
        
        - Usage 
        
```
singularity exec gen_metadata_v2.sif gen_metadata.py INHS_FISH_50577.jpg result_metadata.json mask.png
or
singularity exec gen_metadata_v2.sif gen_metadata.py INHS_FISH_50577.jpg result_metadata.json
```
        
        
