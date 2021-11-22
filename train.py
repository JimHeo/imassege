import dcnn_module.config as config
from dcnn_module.dataset import SegmentationDataset
from dcnn_module.neural_network.mini_unet import UNet
from dcnn_module.utils.metrics_torch import Accuracy, F1Score, IoU, BinaryFocalLoss, CategoricalFocalLoss
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
from sklearn.model_selection import train_test_split
from math import ceil
from imutils import paths
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt
import torch
import time
import os

# load the image and mask filepaths in a sorted manner
image_paths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
mask_paths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

# partition the data into training and testing splits
# using 80% of the data for training and the remaining 20% for testing
if config.TEST_SPLIT:
    split = train_test_split(image_paths, mask_paths, test_size=config.TEST_SPLIT, random_state=42)
    # unpack the data split
    (train_images, test_images) = split[:2]
    (train_masks, test_masks) = split[2:]
else:
    (train_images, test_images) = (config.TRAIN_IMAGES, config.TEST_IMAGES)
    (train_masks, test_masks) = (config.TRAIN_MASKS, config.TEST_MASKS)

# write the testing image paths to disk so that we can use then
# when evaluating/testing our model
print("[INFO] saving testing image paths...")
with open(config.TEST_PATHS, "w") as f:
    f.write("\n".join(test_images))
    
transforms = A.Compose([
    A.HorizontalFlip(p=0.25),
    A.ShiftScaleRotate(p=0.25),
    A.GaussNoise(p=0.25),
    A.RandomBrightnessContrast(p=0.25),
    A.OneOf([A.augmentations.geometric.transforms.ElasticTransform(p=0.25),
             A.augmentations.geometric.transforms.Perspective(p=0.25),
             A.augmentations.geometric.transforms.PiecewiseAffine(p=0.25),
             A.OpticalDistortion(p=0.25)], p=0.25),
])

# create the train and test datasets
patch_size = (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)
training_set = SegmentationDataset(image_paths=train_images, mask_paths=train_masks,
                                   resize=None, random_crop=patch_size, normalization="z-score",
                                   channels=config.INPUT_CHANNEL, classes=config.NUM_CLASSES,
                                   transforms=transforms)
test_set = SegmentationDataset(image_paths=test_images, mask_paths=test_masks,
                               resize=None, random_crop=patch_size, normalization="z-score",
                               channels=config.INPUT_CHANNEL, classes=config.NUM_CLASSES,
                               transforms=None)

print(f"[INFO] found {len(training_set)} examples in the training set...")
print(f"[INFO] found {len(test_set)} examples in the test set...")
# create the training and test data loaders
train_loader = DataLoader(training_set, shuffle=True,
                         batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                         num_workers=os.cpu_count())
test_loader = DataLoader(test_set, shuffle=False,
                        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                        num_workers=os.cpu_count())

# initialize Neural Network model
model = UNet(input_channel=config.INPUT_CHANNEL, num_classes=config.NUM_CLASSES, downsampling="strided", upsampling="bilinear").to(config.DEVICE)
# summary(model, (config.BATCH_SIZE, config.INPUT_CHANNEL, config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH))

# initialize loss function and optimizer
if config.NUM_CLASSES == 1: loss_func = BCEWithLogitsLoss()
else: loss_func = CrossEntropyLoss()
# if config.NUM_CLASSES == 1: loss_func = BinaryFocalLoss(alpha=0.25, gamma=2.)
# else: loss_func = CategoricalFocalLoss(num_classes=config.NUM_CLASSES, alpha=[0.25, 0.75], gamma=2.)
metrics = [Accuracy(config.NUM_CLASSES), F1Score(config.NUM_CLASSES), IoU(config.NUM_CLASSES)]
opt = Adam(model.parameters(), lr=config.INIT_LR)

# calculate steps per epoch for training and test set
train_steps = ceil(len(training_set) / config.BATCH_SIZE)
test_steps = ceil(len(test_set) / config.BATCH_SIZE)

# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}
for metric in metrics:
    H["train_" + str(metric)] = []
    H["test_" + str(metric)] = []

# loop over epochs
print("[INFO] training the network...")
start_time = time.time()
for epoch in tqdm(range(config.NUM_EPOCHS)):
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss
    total_train_loss = 0.
    total_test_loss = 0.
    total_train_metrics = [0. for _ in range(len(metrics))]
    total_test_metrics = [0. for _ in range(len(metrics))]

    # loop over the training set
    for (x, y) in train_loader:
        # send the input to the device
        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = loss_func(pred, y)
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far
        total_train_loss += loss
        for i in range(len(metrics)):
            total_train_metrics[i] += metrics[i](pred, y).cpu().detach().numpy()
            
    # switch off autograd
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (x, y) in test_loader:
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            # make the predictions and calculate the validation loss
            pred = model(x)
            total_test_loss += loss_func(pred, y)
            for i in range(len(metrics)):
                total_test_metrics[i] += metrics[i](pred, y).cpu().detach().numpy()
                
    # calculate the average training and validation loss
    avg_train_loss = total_train_loss / train_steps
    avg_test_loss = total_test_loss / test_steps
    avg_train_metrics = []
    avg_test_metrics = []
    for total_train_metric, total_test_metric in zip(total_train_metrics, total_test_metrics):
        avg_train_metrics.append(total_train_metric / train_steps)
        avg_test_metrics.append(total_test_metric / test_steps)
        
    # update our training history
    H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
    H["test_loss"].append(avg_test_loss.cpu().detach().numpy())
    for metric, avg_train_metric in zip(metrics, avg_train_metrics):
        H["train_" + str(metric)].append(avg_train_metric)
    for metric, avg_test_metric in zip(metrics, avg_test_metrics):
        H["test_" + str(metric)].append(avg_test_metric)
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(epoch + 1, config.NUM_EPOCHS))
    print("Train loss: {:.6f}, Test loss: {:.4f}".format(avg_train_loss, avg_test_loss))
    for metric, avg_train_metric, avg_test_metric in zip(metrics, avg_train_metrics, avg_test_metrics):
        print("Train {}: {:.6f}, Test {}: {:.6f}".format(str(metric), avg_train_metric, str(metric), avg_test_metric))
# display the total time needed to perform the training
end_time = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(end_time - start_time))

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(os.path.join(config.BASE_OUTPUT, "loss.png"))

for metric in metrics:
    # plot the training and test metrics
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_" + str(metric)], label="train_" + str(metric))
    plt.plot(H["test_" + str(metric)], label="test_" + str(metric))
    plt.title("Training metric on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Scores")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(config.BASE_OUTPUT, str(metric) + ".png"))

# serialize the model to disk
torch.save(model.state_dict(), config.MODEL_PATH)
