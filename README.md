## Overall Pipeline

0. Review data for adherence to quality standards, if multi-layered TIFFs run export_top_layer.py
1. Split large images into regular TIFF format (from BigTIFF, run large_image_splitter.py and gimp convert-tiff script)
2. a. Process image through QuPath to obtain cell locations and clustering information
2. b. Create image binary masks through gimp (gimp_batch_mask)
3. Process images and features into sequences as binary files (preprocess_lstm.py)
4. Run networks (training or testing, recurrence_seq_lstm.py)
5. Summarize training conditions (ID_post_processing.py), or summarize a single testing condition (majority_vote.py)
6. Summarize across multiple tests (multitest_summary.py)



## preprocess_lstm.py

Append together image binary files for feeding into recurrence_seq_lstm. If an image's binary files does not exist, then one will first be created and then appended to the condition binary file.

If a binary file does not exist, and some of the necessary data for its creation is unavailable, then it will be skipped and omitted from the condition binary file.

This script inherently depends on centroid_lstm.py and qupath_lstm.py. With certain parameters, it can additionally interact with a subject_list_generator.py and gimp_threshold_mask (a custom gimp batch script).

### --condition_path

Path of directory with condition folders containing lists of images. *This should be a complete file path.*

The directory specified by this parameter should contain subdirectories of cross-validation conditions named of the convention:
001_condition
002_condition
...
00n_condition

Where *n* is the total number of cross-validation conditions.
```
python3 preprocess_lstm.py --mode=/data/recurrence_seq_lstm/data_conditions/Big_Institution_9
```

### --recurrence_only

If the **condition_path** is specified, this flag is ignored.

Generate only recurrence binary file from lists in the config.image_data_folder_path/per_mode_subjects directory.

```
python3 preprocess_lstm.py --recurrence_only
```

### --nonrecurrence_only

If the **condition_path** is specified, this flag is ignored.

Generate only nonrecurrence binary file from lists in the config.image_data_folder_path/per_mode_subjects directory.

```
python3 preprocess_lstm.py --nonrecurrence_only
```

### --seg_path

Segmentations are custom-made binary masks for sampling from specific areas within the image.

This parameter specifies the directory containing binary segmentations.

```
python3 preprocess_lstm.py --seg_path=/data/recurrence_seq_lstm/segmentations/
```

### --mode

Create the condition binary file for a specific model 'mode' type. Allowable modes are 'train', 'valid', or 'test'.

```
python3 preprocess_lstm.py --mode=valid
```



### And coding style tests

Explain what these tests test and why

```
Give an example
```
