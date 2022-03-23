singularity pull library://thibaulttabarin/bgnn/detectron2_env
singularity exec detectron2_env.sif python gen_metadata.py INHS_FISH_50577.jpg result.json
