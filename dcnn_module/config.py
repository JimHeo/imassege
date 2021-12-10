# import the necessary packages
import torch
import os
from dcnn_module.dataset_module.connection import DatasetParser

# split data of the dataset
# current implementation: "BlurDataset", "TGS_Salt"
DATASET_NAME = "BlurDataset"
DATA_SPLIT = 0 # 0.2
TRAIN_LIST_FILE = "train_list.txt" # None
VALID_LIST_FILE = "test_list.txt" # None
DATASET_PARSER = DatasetParser(DATASET_NAME, "dataset")
TRAIN_IMAGES, TRAIN_MASKS, VALID_IMAGES, VALID_MASKS = DATASET_PARSER.connection(split=DATA_SPLIT, train_list_file=TRAIN_LIST_FILE, valid_list_file=VALID_LIST_FILE)

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
INPUT_CHANNEL = 3
NUM_CLASSES = 1

# define the preprocessing parameters
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256

# initialize learning rate, number of epochs to train for, and the batch size
INIT_LR = 0.0001
NUM_EPOCHS = 50
BATCH_SIZE = 16

# determine the loss function multi or single
# using pretrained model
MULTI_LOSS = True
PRETRAINED = True

# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training
# plot, and testing image paths
PREDICTION_OUTPUT = os.path.join(BASE_OUTPUT, "predictions")