
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

A configuration must be created prior to running the main directory code. Run this code, follow the prompts, and move your data to the correct locations prior to running `process.py` or `vote.py`.


## process.py 
Requires a valid image, tsv of qupath output, binary mask from gimp, and image_list.csv. 

```
python process.py
```

If using a custom configuration, specify with the `--conf` argument (the only argument):
```
python process.py --conf=custom_config.file
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

### Outputs
If an image binary file does not exist, it is created and saved to the `image_bin_dir` directory specified in the configuration

All issues encountered in creating image binaries are detailed in the `err_csv` specified in the configuration (error\_list.csv by default).

Six condition binary files are created for running recurrence_lstm:
* recurrence_train.bin
* recurrence_valid.bin
* recurrence_test.bin
* nonrecurrence_train.bin
* nonrecurrence_valid.bin
* nonrecurrence_test.bin

If no images are listed for a given mode, or no image binary files are successfully found/created, then an empty binary file is created. This is acceptable behavior for the testing condition where the train and valid files are required to exist, but are not used.

## recurrence\_lstm\_features.py and recurrence\_lstm\_FC\_features.py
Using the binary files created by process.py, train a the neural network into a set of usable parameters

Whenever running this script, the `--name` parameter is required. Name should be a unique string identifying the current training or testing session.

To load data from a non-default location, use the `--data_path` parameter:
```
python recurrence_lstm_features.py --name=[DNN training name] --data_path=[str filepath to data]
```

### Training
By default, the network runs the training configuration for the number of epochs specified in the config file. Training can be started with the line:
```
python recurrence_lstm_features.py --name=[DNN training name]
```

Default training uses the hyperparameters `--learning_rate=0.005` and `--keep_prob=0.7` (a dropout rate of 0.3). These hyperparameters can be changed at the command line:
```
python recurrence_lstm_features.py --name=[DNN training name] --learning_rate=1e-5 --keep_prob=0.9
```

### Testing
To test a trained model, set the `--config` and `--model_path` parameters:
```
python recurrence_lstm_features.py --name=[DNN testing name] --config=test --model_path=[filepath to model directory]
```

**Note:** For the model being tested, the model_path directory must contain the `checkpoint` file with lines accurately pointing to the `.data`, `.index`, and `.meta` checkpoint files.

## vote.py
Process output of recurrence_lstm into voting scores for all subjects in a testing condition. Requires specifying the model results you want to summarize with the `--model` command-line argument:
```
python vote.py --model=[DNN testing name]
```

The value of `--model` must be a directory in the configuration's `results_dir`. If using a custom configuration, specify with the `--conf` argument.

## summary.py
Process the training logs of recurrence_lstm into a figure saved as a jpg image. The figure is a 2x2 subplot with values of each available epoch for:

* Specificity
* Sensitivity
* Accuracy
* Loss

Requires specifying the model results you want to summarize with the `--model` command-line argument:
```
python summary.py --model=[DNN testing name]
```
