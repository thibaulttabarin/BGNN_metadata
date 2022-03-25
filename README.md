# Drexel Metadata

## Goal
To develop a tool to check the validity of metadata associated with an image, and generate things that are missing. Also includes various geometric and statistical properties on the mask generated over the biological specimen presented.

## Functionality

Object detection is currently being performed on 5 detection classes (fish, fish eyes, rulers, and the twos and threes found on rulers). The current setup is performed on the INHS and UWZM biological specimen image repositories.

### Current Criteria

1. Image must contain a fish species (no eels, seashells, butterflies, seahorses,
snakes, etc).
2. Image must contain only 1 of each class (except eyes).
3. Specimen body must lie alone the image plane from a side view.
4. Ruler must be consistent (only two ruler types, INHS + UWZM, were used
in training set).
5. Fish must not be obscured by another object (petri dish for example).
6. Whole body of fish must be present (no heads, tails, or standalone features).
7. Fish body must not be folded and should have no curvature.

These do not need to be adhered to if properly set up/modified for a specific use case.

### Dependencies

Every dependency is stored in a Pipfile. To set this up, run the following commands:

```bash
pip install pipenv
pipenv install
```

There may be OS dependent installations one may need to perform.

### Training

Setup:

1. Create a COCO JSON training set using the images and labels.
    1. This is done currently using [makesense](https://makesense.ai) in their polygon object detection mode.
    2. The labels currently used are: `fish, ruler, eye, two, three` in that exact order.
    3. Save as a COCO JSON after performing manual segmentations and labeling. Then, place in [datasets](datasets/).
2. In the [config](config/) directory, create a JSON file with a key name of the image directory on your local system, and then a value of an array of dataset names in the [datasets](datasets/) folder.
    1. For multiple image collections, have multiple keys.
    2. For multiple datasets for the same collection, append to the respective value array.
    3. For example: `{"image_folder_name": ["dataset_name.json"]}`.
3. In the [train](train_model.py) script, set the `conf` variable at the top of the `main()` function to load the JSON file name created in the previous step.
4. Create a text file named `overall_prefix.txt` file in the [config](config/) folder. This file should have the absolute path to the directory in which all the image repositories will be stored.
    1. Currently it is `/usr/local/bgnn/`. There are various image folders like `tulane`, `inhs_filtered`, `uwzm_filtered`, etc.
5. To edit the learning rate, batch size, or any other base training configuration, edit the [base training configurations](config/Base-RCNN-FPN.yaml) file. 
6. To edit the number of iterations, dropoffs, or any model specific configurations, edit the [model training configurations](config/mask_rcnn_R_50_FPN_3x.yaml) file.

Finally, to train the model, run the following command:

```bash
pipenv run python3 train_model.py
```

## Metadata Generation

The metadata generated is extremely specific to our use case. In addition, we perform additional image processing techniques to improve our accuracies that may not work for other use cases. These include:

1. Image scaling when a fish is detected but not an eye, in an attempt to lower missing eyes.
2. Selection of highest confidence fish bounding box given our criterion of single fish in an image.
3. Contrast enhancement (CLAHE)

The metadata generated produces various statistical and geometric properties of a biological specimen image or collection in a JSON format. When a single file is passed, the data is yielded to the console (stdout). When a directory is passed, the data is stored in a JSON file.

To generate the metadata, run the following command:
```bash
pipenv run python3 gen_metadata.py [file_or_dir_name]
```

## Properties Generated

| **Property**            | **Association** | **Type** | **Explanation**                                                                                                                                   |
|----------------------------------|--------------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| has\_fish               | Overall Image            | Boolean           | Whether a fish was found in the image.                                                                                                                      |
| fish\_count             | Overall Image            | Integer           | The quantity of fish present.                                                                                                                               |
| has\_ruler              | Overall Image            | Boolean           | Whether a ruler was found in the image.                                                                                                                     |
| ruler\_bbox             | Overall Image            | 4 Tuple           | The bounding box of the ruler (if found).                                                                                                                   |
| scale*                  | Overall Image            | Float             | The scale of the image in $\frac{\mathrm{pixels}}{\mathrm{cm}}$.                                                                                          |
| bbox                    | Per Fish                 | 4 Tuple           | The top left and bottom right coordinates of the bounding box for a fish.                                                                                   |
| background.mean         | Per Fish                 | Float             | The mean intensity of the background within a given fish's bounding box.                                                                                    |
| background.std          | Per Fish                 | Float             | The standard deviation of the background within a given fish's bounding box.                                                                                |
| foreground.mean         | Per Fish                 | Float             | The mean intensity of the foreground within a given fish's bounding box.                                                                                    |
| foreground.std          | Per Fish                 | Float             | The standard deviation of the foreground within a given fish's bounding box.                                                                                |
| contrast*               | Per Fish                 | Float             | The contrast between foreground and background intensities within a given fish's bounding box.                                                              |
| centroid                | Per Fish                 | 4 Tuple           | The centroid of a given fish's bitmask.                                                                                                                     |
| primary\_axis*          | Per Fish                 | 2D Vector         | The unit length primary axis (eigenvector) for the bitmask of a given fish.                                                                                 |
| clock\_value*           | Per Fish                 | Integer           | Fish's primary axis converted into an integer "clock value" between 1 and 12.                                                                             |
| oriented\_length*       | Per Fish                 | Float             | The length of the fish bounding box in centimeters.                                                                                                         |
| mask                    | Per Fish                 | 2D Matrix         | The bitmask of a fish in 0's and 1's.                                                                                                                       |
| pixel\_analysis\_failed | Per Fish                 | Boolean           | Whether the pixel analysis process failed for a given fish. If true, detectron's mask and bounding box were used for metadata generation. |
| score                   | Per Fish                 | Float             | The percent confidence score output by detectron for a given fish.                                                                                 |
| has\_eye                | Per Fish                 | Boolean           | Whether an eye was found for a given fish.                                                                                                                  |
| eye\_center             | Per Fish                 | 2 Tuple           | The centroid of a fish's eye.                                                                                                                               |
| side*                   | Per Fish                 | String            | The side (i.e. 'left' or 'right') of the fish that is facing the camera (dependent on finding its eye).                                  |
| area                 | Per Fish                 | Float             | Area of fish in $\mathrm{cm^2}$.                                                                                               |
| cont\_length         | Per Fish                 | Float             | The longest contiguous length of the fish in centimeters.                                                                        |
| cont\_width          | Per Fish                 | Float             | The longest contiguous width of the fish in centimeters.                                                                         |
| convex\_area         | Per Fish                 | Float             | Area of convex hull image (smallest convex polygon that encloses the fish) in $\mathrm{cm^2}$.                                 |
| eccentricity         | Per Fish                 | Float             | Ratio of the focal distance over the major axis length of the ellipse that has the same second-moments as the fish.              |
| extent               | Per Fish                 | Float             | Ratio of pixels of fish to pixels in the total bounding box. Computed as $\frac{\mathrm{area}}{\mathrm{rows} * \mathrm{cols}}$ |
| feret\_diameter\_max | Per Fish                 | Float             | The longest distance between points around the fish’s convex hull contour.                                                       |
| kurtosis             | Per Fish                 | 2D Vector         | The sharpness of the peaks of the frequency-distribution curve of mask pixel coordinates.                                        |
| major\_axis\_length  | Per Fish                 | Float             | The length of the major axis of the ellipse that has the same normalized second central moments as the fish.                     |
| mask.encoding        | Per Fish                 | String            | The 8-way Freeman Encoding of the outline of the fish.                                                                           |
| mask.start\_coord    | Per Fish                 | 2D Vector         | The starting coordinate of the Freeman encoded mask.                                                                             |
| minor\_axis\_length  | Per Fish                 | Float             | The length of the minor axis of the ellipse that has the same normalized second central moments as the fish.                     |
| oriented\_width      | Per Fish                 | Float             | The width of the fish bounding box in centimeters.                                                                               |
| perimeter            | Per Fish                 | Float             | The approximation of the contour in centimeters as a line through the centers of border pixels using 8-connectivity.             |
| skew                 | Per Fish                 | 2D Vector         | The measure of asymmetry of the frequency-distribution curve of mask pixel coordinates.                                          |
| solidity             | Per Fish                 | Float             | The ratio of pixels in the fish to pixels of the convex hull image.                                                              |
| std             | Per Fish                 | Float             | The standard deviation of the mask pixel coordinate distribution. |

## Authors

Joel Pepper

Kevin Karnani

