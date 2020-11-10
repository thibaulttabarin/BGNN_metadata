# drexel_metadata
## Status
* `gen_metadata.py`: Just some small pixel analysis
* `train_detectron.py`: Does the following:
    * Reads in nrrd files and creates a [COCO](https://cocodataset.org/#home) format dataset for use by [detectron2](https://github.com/facebookresearch/detectron2).
    * Trains an image detection model on the dataset
    * While it makes a validation dataset, I still need to configure the script on how to use it so no proper validation is happening yet.
* `old.py`: Contains some data visualization code for the nrrds that I am not currently using but didn't want to delete.
* `test_model.py`: Runs a trained model on a random sample of INHS images and outputs the detection results as image overlays.
