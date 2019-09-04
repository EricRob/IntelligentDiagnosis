
# Environment setup
## Docker

First, install [docker](https://docs.docker.com/install/#server) and [docker-compose](https://docs.docker.com/compose/install/).

__docker-compose must have minimum version 1.19.0 to use nvidia runtime__

Next, make sure the data you want to access from the container is in the directories specified in the `volumes:` section of `docker-compose.yml`. Ideally, attach all local volumes to the `/opt/out` directory in the docker container.

To create the container:
```
sudo docker-compose build
```

__When there is a change to `docker-compose.yml` or `build/mela-nn.Dockerfile`, the container must be rebuilt.__

To start the built container and open into a bash prompt:
```
sudo docker-compose run mela-nn bash
```

To attach a volume that is not available by default, add the `-v` flag:
```
sudo docker-compose -v [local dir]:[container dir] run mela-nn bash
```

__ATTN:__ _Use the code in the main directory unless you want a specific, known feature from the leagacy code_

# Running The Code
## config.py
A configuration must be created prior to running the main directory code. 
Run this code, follow the prompts asking if you'd like to create a new configuration or overwrite the current default configuration. 

```
python config.py
```

Move your data to the correct locations specified in config.py prior to running `process.py` or `vote.py`.

## process.py 
Requires a valid image, tsv of qupath output, binary mask from gimp, and image_list.csv. 

```
python process.py
```

If using a custom configuration, specify with the `--conf` argument:
```
python process.py --conf=custom_config.file
```

By default process.py will create all train, valid, and test sets. If you want to create only a certain set, specify with the `--set` argument:
```
python process.py --set=train
```

By default process.py will create all recurrence and nonrecurrence binaries. If you want to process only one, specify with the `--label` argument
```
python process.py --label=recurrence
```

If want to create binary files for an additional set of images not in the default image_list, specify the `--image_list` argument (Must specify entire path and must be a csv file):
```
python process.py --image_list=./data/testing_images.csv
```

### Sample image_list.csv:

|  mode  | subject |    image     | label | source |
|--------|---------|--------------|-------|--------|
| train  |  00-00  | 00_00_00.tif |   0   |  CUMC  |
| train  |  11-11  | 11_11_11.tif |   1   |  CUMC  |
| valid  |  00-00  | 00_00_00.tif |   0   |  CUMC  |
| valid  |  11-11  | 11_11_11.tif |   1   |  CUMC  |
| test   |  00-00  | 00_00_00.tif |   0   |  CUMC  |
| test   |  11-11  | 11_11_11.tif |   1   |  CUMC  |

### Output of running process.py
If an image binary file does not exist, it is created and saved to the 
`image_bin_dir` directory specified in the configuration. All issue
encountered in creating image binaries are detailed in the `err_csv` specified in the configuration (error\_list.csv by default).

Depending on if the set hyperparameter was specified, a maximum of six condition binary files can be created for running recurrence_lstm:
* recurrence_train.bin
* recurrence_valid.bin
* recurrence_test.bin
* nonrecurrence_train.bin
* nonrecurrence_valid.bin
* nonrecurrence_test.bin

Once the image binary file is created (or if it already exists), it's appended to the end of the appropriate condition binary file. This means that if images `A`, `B`, `C`, `D`, and `E` are the recurrent images in the `train` condition, then `recurrence_train.bin` will be equal to `ABCDE`.

If no images are listed for a given mode, or no image binary files are successfully found/created, then an empty binary file is created. This is acceptable behavior for the testing condition where the train and valid files are required to exist but not used.

## recurrence\_lstm\_features.py and recurrence\_lstm\_FC\_features.py
Using the binary files created by process.py, train a the neural network into a set of usable parameters

Whenever running this script, the `--name` parameter is required. Name should be a unique string identifying the current training or testing session.

The default data_path location is in the image_binaries directory. 
To load data from a non-default location, use the `--data_path` parameter:
```
python recurrence_lstm_features.py --name=[DNN training name] --data_path=[str filepath to data]
```

### Training
By default, the network runs the training configuration for the number of epochs specified in the config file. Training can be started with the line:
```
python recurrence_lstm_features.py --name=[DNN training name]
```

Default training uses the hyperparameters `--learning_rate=0.005` and 
`--keep_prob=0.7` (a dropout rate of 0.3). These hyperparameters can be changed at the command line:
```
python recurrence_lstm_features.py --name=[DNN training name] --learning_rate=1e-5 --keep_prob=0.9
```

To specify the number of epochs, use the hyperparameter `--epochs`
```
python recurrence_lstm_features.py --name=[DNN training name] --epochs=50
```

### Testing
If you want to test on a different set of images than the set used in the training data, run this line using the `--image_list` hyperparameter to specify the set of images (this file must be a csv and the entire data path must be specified) and set the `--set` hyperparameter to test:
```
python process.py --set=test --image_list=./data/testing_images.csv
```

To test a trained model, set the `--config` and `--model_path` parameters:
```
python recurrence_lstm_features.py --name=[DNN testing name] --config=test --model_path=[filepath to model directory]
```

**Note:** For the model being tested, the model_path directory must contain the `checkpoint` file with lines accurately pointing to the `.data`, `.index`, and `.meta` checkpoint files.

### Output of testing 
The DNN testing directory can be found in the results directory and contains two files:
* voting_file.csv
* secondary_test_results.txt

## vote.py
Process output of recurrence_lstm into voting scores for all subjects in a testing condition. Requires specifying the model results you want to summarize with the `--model` command-line argument. The value of `--model` must be a directory in the configuration's `results_dir`.
```
python vote.py --model=[DNN testing name]
```

 If using a custom configuration, specify with the `--conf` argument.
```
python vote.py --model=[DNN testing name] --conf=custom_config.file
```

## summary.py
Process the training logs of recurrence_lstm for train, valid, and test into a figure saved as a jpg image. The figure is a 2x2 subplot with values of each available epoch for:

* Specificity
* Sensitivity
* Accuracy
* Loss

Requires specifying the directory with the model results you want to summarize with the `--model` command-line argument. Model directory must contain train_results.txt, test_results.txt, and valid_results.txt because this data is used to generate the summary graphs.

```
python summary.py --model=[DNN training name]
```
