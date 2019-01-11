# Overall Pipeline

0. Review data for adherence to quality standards, if multi-layered TIFFs run export_top_layer.py
1. Split large images into regular TIFF format (from BigTIFF, run large_image_splitter.py and gimp convert-tiff script)
2. a. Process image through QuPath to obtain cell locations and clustering information
2. b. Create image binary masks through gimp (gimp_batch_mask)
3. Process images and features into sequences as binary files (preprocess_lstm.py)
4. Run networks (training or testing, recurrence_seq_lstm.py)
5. Summarize training conditions (ID_post_processing.py), or summarize a single testing condition (majority_vote.py)
6. Summarize across multiple tests (multitest_summary.py)



# preprocess_lstm.py

Append together image binary files for feeding into recurrence_seq_lstm. If an image's binary files does not exist, then one will first be created and then appended to the condition binary file.

If a binary file does not exist, and some of the necessary data for its creation is unavailable, then it will be skipped and omitted from the condition binary file.

This script inherently depends on **centroid_lstm.py** and **qupath_lstm.py**. With certain parameters, it can additionally interact with **subject_list_generator.py** and **gimp_threshold_mask** (a custom gimp batch script).

## Overall File Creation Flags

### --condition_path

Path of directory with condition folders containing lists of images. *This should be a complete file path.*

The directory specified by this parameter should contain subdirectories of cross-validation conditions named of the convention:
001_condition \
002_condition \
... \
00n_condition \

Where *n* is the total number of cross-validation conditions.

Default: *None*


```
python3 preprocess_lstm.py --condition_path=/data/recurrence_seq_lstm/data_conditions/Big_Institution_9
```

### --generate_conditions

Generate randomized lists of subjects for cross-validation testing. Succesfully using this parameter also requires specifying **--verified_list** and **--new_condition** parameters.

Default: *None*


```
python3 preprocess_lstm.py --generate_conditions --verified_list=NewSubjList.csv --new_condition=New_Subjects_Condition
```

### --verified_list

List of subjects used for creating new cross-validation conditions. It is assume that this list exists in the folder specified by **--data_conditions** Succesfully using this parameter also requires specifying **--generate_conditions** and **--new_condition** parameters.

Default: *None*


```
python3 preprocess_lstm.py --generate_conditions --verified_list=NewSubjList.csv --new_condition=New_Subjects_Condition
```

### --cross_val_folds

Number of folds for the new cross-validation conditions. Typically, this parameter does not change (in this code base, the default is 6). It is likely easiest to manually change the default value in the code and avoid needing to set this.

Succesfully using this parameter also requires specifying **--generate_conditions**, **--verified_list**, and **--new_condition** parameters.

Default: *6*


```
python3 preprocess_lstm.py --generate_conditions --cross_val_folds=6 --verified_list=NewSubjList.csv --new_condition=New_Subjects_Condition
```

### --data_conditions

Location of data_conditions directory, aka parent directory for the new cross-validation condition folder. This should be set once to a default value and never changed.

Succesfully using this parameter also requires specifying **--generate_conditions**, **--verified_list**, and **--new_condition** parameters.

Default: */data/recurrence_seq_lstm/data_conditions*

```
python3 preprocess_lstm.py --generate_conditions --data_conditions=/data/recurennce_seq_lstm/data_conditions/ --verified_list=NewSubjList.csv --new_condition=New_Subjects_Condition
```

### --config

Configuration for generating patches. Configurations can be added as needed, but only through changing the code. The following configurations exist:
* "original" used for CUMC, and Geisinger 40x testing and training
* "small" used for smaller-patch testing of 40x samples
* "yale" used for 20x images from Yale
* "yale_small" used for smaller-patch testing of 20x samples
* "sinai" used for 40x images from Sinai (different sampling parameters than "original)

If an invalid configuration is specified then the "original" config is used.

Default: *original*

```
python3 preprocess_lstm.py --config=original
```

### --sampling_method

Sampling pattern for generating sequences. Three methods currently exist: gauss, column, and row

Default: *gauss*

```
python3 preprocess_lstm.py --sampling_method=gauss
```

### --check_new_masks

*ATTN: Do not use this parameter before thoroughly reading through the code. If any part of the parameters are off then it can cause a real mess of things and potentially overwrite data.*

Check for new masks in the new_image_dir directory that need to be created. This allows a method to run that triggers the batch-mask gimp script.

This should be run with specifying **--new_image_dir**, **--original_image_dir**, and **--masks_dir** unless the default for those parameters have been changed.

Default: *False*

```
python3 preprocess_lstm.py --check_new_masks --new_image_dir=/home/Desktop/new_masks/
```

### --new_image_dir

Location of new images to be masked and moved to the directory specified in **--original_image_dir**. Masks are moved to **--masks_dir**

Used together with **--check_new_masks**, **--original_images_dir**, and **--masks_dir**

Default: */home/wanglab/Desktop/new_masks*

```
python3 preprocess_lstm.py --check_new_masks  --new_image_dir=/home/Desktop/new_masks/ --original_image_dir=/data/recurrence_seq_lstm/image_data/original_images --masks_dir=/data/recurrence_seq_lstm/image_data/masks/
```

### --original_images_dir

Directory of all original images. This is the new image output after the subprocesses called for binary masking. Note to developer: could be replaced with the value in the configuration.

Used together with **--check_new_masks**, **--new_image_dir**, and **--masks_dir**

Default: */data/recurrence_seq_lstm/image_data/original_images*

```
python3 preprocess_lstm.py --check_new_masks  --new_image_dir=/home/Desktop/new_masks/ --original_image_dir=/data/recurrence_seq_lstm/image_data/original_images --masks_dir=/data/recurrence_seq_lstm/image_data/masks/
```

### --masks_dir

Directory of all mask images. This is the destination for the output of the subprocesses called for binary masking. 

Used together with **--check_new_masks**, **--new_image_dir**, and **--original_images_dir**

Default: */data/recurrence_seq_lstm/image_data/masks*

```
python3 preprocess_lstm.py --check_new_masks  --new_image_dir=/home/Desktop/new_masks/ --original_image_dir=/data/recurrence_seq_lstm/image_data/original_images --masks_dir=/data/recurrence_seq_lstm/image_data/masks/
```

### --preverify

Run preverification to ensure that all images and masks exist that will be used for binary file creation.

Default: *False*

```
python3 preprocess_lstm.py --preverify
```

### --overwrite

Always create new binary files and overwrite existing ones. Use this parameter when some aspect of binary file creation has changed, or the original data used for the binary file is updated.

Default: *False*

```
python3 preprocess_lstm.py --overwrite
```

### --no_write

Skip writing binary files. All normal functions of writing will execute, except the final step of writing to the binary file.

Default: *False*

```
python3 preprocess_lstm.py --no_write
```

### --recurrence_only

If the **--condition_path** is specified, this flag is ignored.

Generate only recurrence binary file from lists in the config.image_data_folder_path/per_mode_subjects directory.

Default: *False*

```
python3 preprocess_lstm.py --recurrence_only
```

### --nonrecurrence_only

If the **--condition_path** is specified, this flag is ignored.

Generate only nonrecurrence binary file from lists in the config.image_data_folder_path/per_mode_subjects directory.

Default: *False*

```
python3 preprocess_lstm.py --nonrecurrence_only
```

### --seg_path

Segmentations are custom-made binary masks for sampling from specific areas within the image.

This parameter specifies the directory containing binary segmentations.

Default: *None*

```
python3 preprocess_lstm.py --seg_path=/data/recurrence_seq_lstm/segmentations/
```

### --mode

Create the condition binary file for a specific model 'mode' type. Allowable modes are 'train', 'valid', or 'test'.

Default: *None*

```
python3 preprocess_lstm.py --mode=valid
```

## Gauss Configuration Flags

### --gauss_seq

Number of sequences to generate per tile with gaussian samples.

**Only works with the "original" configuration.** This should normally be specified in the configuration rather than with this flag.

Default: *6*

```
python3 preprocess_lstm.py --gauss_seq=6
```

### --gauss_stdev

Size of standard deviation from tile centers for 2D gaussian sampling of patch locations.

**Only works with the "original" configuration.** This should normally be specified in the configuration rather than with this flag.

Default: *1500*

```
python3 preprocess_lstm.py --gauss_stdev=1500
```

### --gauss_tile_size

Tile dimensions for splitting sample image for gauss distribution.

**Only works with the "original" configuration.** This should normally be specified in the configuration rather than with this flag.

Default: *1500*

```
python3 preprocess_lstm.py --gauss_tile_size=1500
```
### --gauss_tile_size

Tile dimensions for splitting sample image for gauss distribution.


Default: *1500*

```
python3 preprocess_lstm.py --gauss_tile_size=1500
```



### And coding style tests

Explain what these tests test and why

```
Give an example
```
