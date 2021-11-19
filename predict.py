import dcnn_module.config as config
from dcnn_module.neural_network.mini_unet import UNet
from dcnn_module.metrics import Accuracy, F1_Score, IOU
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

def prepare_plot(origin, gt_mask, pred_mask, base_name=None):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origin)
    ax[1].imshow(gt_mask)
    ax[2].imshow(pred_mask)
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    output_path = os.path.join(config.PREDICTION_OUTPUT, base_name)
    plt.savefig(output_path)
 
def make_predictions(model, image_path):
    base_name = os.path.basename(image_path)
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        if config.INPUT_CHANNEL == 1:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        origin = image.copy()
        # # resize the image
        image = cv2.resize(image, (config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH))
        # find the filename and generate the path to ground truth
        filename = image_path.split(os.path.sep)[-1]
        gt_path = os.path.join(config.MASK_DATASET_PATH, filename)
        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        # gt_mask = cv2.resize(gt_mask, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))
        
        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        if config.INPUT_CHANNEL == 1: image = image = np.expand_dims(image, 0)
        else: image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        pred_mask = model(image).squeeze()
        if config.NUM_CLASSES == 1:
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask.cpu().numpy()
            pred_mask = (pred_mask > config.THRESHOLD) * 255
        else:
            pred_mask = torch.softmax(pred_mask, dim=0)
            pred_mask = torch.argmax(pred_mask, dim=0)
            pred_mask = pred_mask.cpu().numpy()
            
        pred_mask = pred_mask.astype(np.uint8)
        # filter out the weak predictions and convert them to integers
        pred_mask = cv2.resize(pred_mask, gt_mask.shape, interpolation=cv2.INTER_NEAREST)
        # prepare a plot for visualization
        prepare_plot(origin, gt_mask, pred_mask, base_name)
        
# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
image_paths = open(config.TEST_PATHS).read().strip().split("\n")
image_paths = np.random.choice(image_paths, size=10)
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
model = UNet(input_channel=config.INPUT_CHANNEL, num_classes=config.NUM_CLASSES).to(config.DEVICE)
model.load_state_dict(torch.load(config.MODEL_PATH))
# iterate over the randomly selected test image paths
for path in image_paths:
    # make predictions and visualize the results
    make_predictions(model, path)
    