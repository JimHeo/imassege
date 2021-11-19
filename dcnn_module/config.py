# import the necessary packages
import torch
import os
# base path of the dataset
# DATASET_PATH = os.path.join("dataset/TGS_Salt", "train")
DATASET_PATH = os.path.join("dataset", "BlurDataset")

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "image")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "gt")

# define the test split
# TEST_SPLIT = 0.2
TEST_SPLIT = 0
# or define the train and test list
if not TEST_SPLIT:
    TRAIN_LIST_PATH = os.path.join(DATASET_PATH, "train_list.txt")
    TEST_LIST_PATH = os.path.join(DATASET_PATH, "test_list.txt")
    TRAIN_IMAGES = []
    TRAIN_MASKS = []
    TEST_IMAGES = []
    TEST_MASKS = []
    with open(TRAIN_LIST_PATH, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            parsed_line = line.strip().strip('\'')
            TRAIN_IMAGES.append(os.path.join(IMAGE_DATASET_PATH, parsed_line))
            parsed_line = parsed_line.split('.')[0] + '.png'
            TRAIN_MASKS.append(os.path.join(MASK_DATASET_PATH, parsed_line))
    with open(TEST_LIST_PATH, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            parsed_line = line.strip().strip('\'')
            TEST_IMAGES.append(os.path.join(IMAGE_DATASET_PATH, parsed_line))
            parsed_line = parsed_line.split('.')[0] + '.png'
            TEST_MASKS.append(os.path.join(MASK_DATASET_PATH, parsed_line))

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

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.0001
NUM_EPOCHS = 500
BATCH_SIZE = 64
POOLING_LEVEL = 4

# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training
# plot, and testing image paths
# MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_blur_detection.pth")
TEST_PATHS = os.path.join(BASE_OUTPUT, "test_paths.txt")
PREDICTION_OUTPUT = os.path.join(BASE_OUTPUT, "predictions")