import dcnn_module.config as config
from dcnn_module.neural_network.mini_unet import UNet
from dcnn_module.utils.preprocessing import cropping_to_fit, padding_to_fit, normalize
from dcnn_module.utils.metrics_numpy import Accuracy, F1Score, IoU
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

metrics = [Accuracy(config.NUM_CLASSES), F1Score(config.NUM_CLASSES), IoU(config.NUM_CLASSES)]
total_metrics = [0. for _ in range(len(metrics))]

def prepare_plot(origin, gt_mask, pred_mask, base_name=None):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origin)
    ax[1].imshow(gt_mask, cmap="gray")
    ax[2].imshow(pred_mask, cmap="gray")
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    output_path = os.path.join(config.PREDICTION_OUTPUT, base_name)
    plt.savefig(output_path)
    plt.close()
 
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
        origin = image.copy()
        # resize the image
        if str(model) == 'unet': pooling_level = 4
        elif str(model) == 'mini_unet': pooling_level = 1
        image = cropping_to_fit(image, level=pooling_level)
        image = image.astype(np.float32)
        image = normalize(image, type="z-score")
        
        # find the filename and generate the path to ground truth
        filename = image_path.split(os.path.sep)[-1].split(".")[0] + ".png"
        gt_path = os.path.join(config.MASK_DATASET_PATH, filename)
        # load the ground-truth segmentation mask in grayscale mode
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        # if needed, convert 255 to 1
        gt_mask[gt_mask == 255] = 1
        
        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        if config.INPUT_CHANNEL == 1: image = image = np.expand_dims(image, 0)
        else: image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a numpy array
        pred_mask = model(image).squeeze()
        if config.NUM_CLASSES == 1:
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask.cpu().numpy()
            pred_mask = (pred_mask > config.THRESHOLD).astype(np.uint8)
        else:
            pred_mask = torch.softmax(pred_mask, dim=0)
            pred_mask = pred_mask.cpu().numpy()
            pred_mask = np.transpose(pred_mask, (1, 2, 0))
        
        pred_mask = padding_to_fit(pred_mask, gt_mask.shape)
        for i in range(len(metrics)):
            total_metrics[i] += metrics[i](pred_mask, gt_mask)
        if config.NUM_CLASSES > 1: pred_mask = np.argmax(pred_mask, axis=-1).astype(np.uint8)
        # prepare a plot for visualization
        prepare_plot(origin, gt_mask, pred_mask, base_name)
        
# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
with open(config.TEST_PATHS, "r") as f:
    image_paths = f.read().strip().split("\n")
# image_paths = np.random.choice(image_paths, size=10)

# load our model from disk and flash it to the current device
print("[INFO] load up model...")
model = UNet(input_channel=config.INPUT_CHANNEL, num_classes=config.NUM_CLASSES, downsampling="strided", upsampling="bilinear").to(config.DEVICE)
model.load_state_dict(torch.load(config.MODEL_PATH))
# iterate over the randomly selected test image paths
for path in image_paths:
    # make predictions and visualize the results
    make_predictions(model, path)

avg_metrics = []
for metric, total_metric in zip(metrics, total_metrics):
    avg_metric = total_metric / len(image_paths)
    avg_metrics.append(avg_metric)
    print("Mean {}: {:.4f}".format(str(metric), avg_metric))
