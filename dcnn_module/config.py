# import the necessary packages
import torch
import os
# base path of the dataset
DATASET_NAME = "BlurDataset"
# DATASET_PATH = os.path.join("dataset/TGS_Salt", "train")
DATASET_PATH = os.path.join("dataset", DATASET_NAME)

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "image")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "gt")

# define the split
# SPLIT = 0.2
SPLIT = 0
# or define the train and validation list
if not SPLIT:
    TRAIN_LIST_PATH = os.path.join(DATASET_PATH, "train_list.txt")
    VALID_LIST_PATH = os.path.join(DATASET_PATH, "test_list.txt")
    TRAIN_IMAGES = []
    TRAIN_MASKS = []
    VALID_IMAGES = []
    VALID_MASKS = []
    with open(TRAIN_LIST_PATH, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            parsed_line = line.strip().strip('\'')
            TRAIN_IMAGES.append(os.path.join(IMAGE_DATASET_PATH, parsed_line))
            parsed_line = parsed_line.split('.')[0] + '.png'
            TRAIN_MASKS.append(os.path.join(MASK_DATASET_PATH, parsed_line))
    with open(VALID_LIST_PATH, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            parsed_line = line.strip().strip('\'')
            VALID_IMAGES.append(os.path.join(IMAGE_DATASET_PATH, parsed_line))
            parsed_line = parsed_line.split('.')[0] + '.png'
            VALID_MASKS.append(os.path.join(MASK_DATASET_PATH, parsed_line))
TRAIN_IMAGES = sorted(TRAIN_IMAGES)
TRAIN_MASKS = sorted(TRAIN_MASKS)
VALID_IMAGES = sorted(VALID_IMAGES)
VALID_MASKS = sorted(VALID_MASKS)

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