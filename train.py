import dcnn_module.config as config
from dcnn_module.dataset import SegmentationDataset
from dcnn_module.neural_network.mini_unet import UNet
from dcnn_module.neural_network.deeplab.deeplab import DeepLab
from dcnn_module.utils.metrics_torch import Accuracy, F1Score, IoU, DiceLoss
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary
from sklearn.model_selection import train_test_split
from math import ceil
from imutils import paths
from tqdm import tqdm
import torch.nn.functional as F
import albumentations as A
import matplotlib.pyplot as plt
import torch
import time
import os
import smtplib
from email.mime.text import MIMEText

# load the image and mask filepaths in a sorted manner
image_paths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
mask_paths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

# partition the data into training and validation splits
# using 80% of the data for training and the remaining 20% for validation
if config.SPLIT:
    split = train_test_split(image_paths, mask_paths, test_size=config.SPLIT, random_state=42)
    # unpack the data split
    (train_images, valid_images) = split[:2]
    (train_masks, valid_masks) = split[2:]
else:
    (train_images, valid_images) = (config.TRAIN_IMAGES, config.VALID_IMAGES)
    (train_masks, valid_masks) = (config.TRAIN_MASKS, config.VALID_MASKS)
    
transforms = A.Compose([
    A.HorizontalFlip(p=0.25),
    A.ShiftScaleRotate(p=0.25),
    A.GaussNoise(p=0.25),
    A.augmentations.geometric.transforms.ElasticTransform(p=0.25, alpha=300, sigma=20)
])

# create the train and test datasets
patch_size = (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)
training_set = SegmentationDataset(image_paths=train_images, mask_paths=train_masks,
                                   resize=None, random_crop=patch_size, normalization="z-score",
                                   channels=config.INPUT_CHANNEL, classes=config.NUM_CLASSES,
                                   transforms=transforms)
validation_set = SegmentationDataset(image_paths=valid_images, mask_paths=valid_masks,
                               resize=None, random_crop=patch_size, normalization="z-score",
                               channels=config.INPUT_CHANNEL, classes=config.NUM_CLASSES,
                               transforms=None)

print(f"[INFO] found {len(training_set)} examples in the training set...")
print(f"[INFO] found {len(validation_set)} examples in the test set...")
# create the training and test data loaders
train_loader = DataLoader(training_set, shuffle=True,
                         batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                         num_workers=os.cpu_count())
valid_loader = DataLoader(validation_set, shuffle=False,
                        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                        num_workers=os.cpu_count())

# initialize Neural Network model
# model = UNet(input_channel=config.INPUT_CHANNEL, num_classes=config.NUM_CLASSES, downsampling="strided", upsampling="bilinear").to(config.DEVICE)
model = DeepLab(input_channels=config.INPUT_CHANNEL, num_classes=config.NUM_CLASSES, backbone='resnet', decoder='unet', output_stride=16, pretrained=config.PRETRAINED, multi_loss=config.MULTI_LOSS).to(config.DEVICE)
# summary(model, (config.BATCH_SIZE, config.INPUT_CHANNEL, config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH))

# initialize loss function and optimizer
if config.NUM_CLASSES == 1: loss_func = [BCEWithLogitsLoss(), DiceLoss(config.NUM_CLASSES)]
else: loss_func = [CrossEntropyLoss(), DiceLoss(config.NUM_CLASSES)]

metrics = [Accuracy(config.NUM_CLASSES), F1Score(config.NUM_CLASSES), IoU(config.NUM_CLASSES)]
opt = optim.Adam(model.parameters(), lr=config.INIT_LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.1, patience=30, verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=0, min_lr=1e-08)

# calculate steps per epoch for training and validation set
train_steps = ceil(len(training_set) / config.BATCH_SIZE)
valid_steps = ceil(len(validation_set) / config.BATCH_SIZE)

# initialize a dictionary to store training history
H = {"train_loss": [], "valid_loss": []}
for metric in metrics:
    H["train_" + str(metric)] = []
    H["valid_" + str(metric)] = []

# loop over epochs
print("[INFO] training the network...")
start_time = time.time()
for epoch in tqdm(range(config.NUM_EPOCHS)):
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss
    total_train_loss = 0.
    total_valid_loss = 0.
    total_train_metrics = [0. for _ in range(len(metrics))]
    total_valid_metrics = [0. for _ in range(len(metrics))]

    # loop over the training set
    for (x, y) in train_loader:
        # send the input to the device
        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
        # perform a forward pass and calculate the training loss
        pred = model(x)
        if config.MULTI_LOSS:
            scale_factor = 0.5
            for i, p in enumerate(pred):
                if not i:
                    loss = loss_func[0](p, y) + loss_func[1](p, y)
                else:
                    y_low_scale = F.interpolate(y, scale_factor=scale_factor, mode='bilinear', align_corners=True, recompute_scale_factor=True)
                    loss += loss_func[0](p, y_low_scale) + loss_func[1](p, y_low_scale)
                    scale_factor /= 2
        else:
            loss = loss_func[0](pred, y) + loss_func[1](pred, y)
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far
        total_train_loss += loss
        for i in range(len(metrics)):
            if config.MULTI_LOSS: total_train_metrics[i] += metrics[i](pred[0], y).cpu().detach().numpy()
            else: total_train_metrics[i] += metrics[i](pred, y).cpu().detach().numpy()
            
    # switch off autograd
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (x, y) in valid_loader:
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            # make the predictions and calculate the validation loss
            pred = model(x)
            if config.MULTI_LOSS:
                scale_factor = 0.5
                for i, p in enumerate(pred):
                    if not i:
                        val_loss = loss_func[0](p, y) + loss_func[1](p, y)
                    else:
                        y_low_scale = F.interpolate(y, scale_factor=scale_factor, mode='bilinear', align_corners=True, recompute_scale_factor=True)
                        val_loss += loss_func[0](p, y_low_scale) + loss_func[1](p, y_low_scale)
                        scale_factor /= 2
            else:
                val_loss = loss_func[0](pred, y) + loss_func[1](pred, y)
            total_valid_loss += val_loss
            for i in range(len(metrics)):
                if config.MULTI_LOSS: total_valid_metrics[i] += metrics[i](pred[0], y).cpu().detach().numpy()
                else: total_valid_metrics[i] += metrics[i](pred, y).cpu().detach().numpy()
                
    # calculate the average training and validation loss
    avg_train_loss = total_train_loss / train_steps
    avg_valid_loss = total_valid_loss / valid_steps
    avg_train_metrics = []
    avg_valid_metrics = []
    for total_train_metric, total_valid_metric in zip(total_train_metrics, total_valid_metrics):
        avg_train_metrics.append(total_train_metric / train_steps)
        avg_valid_metrics.append(total_valid_metric / valid_steps)
    
    scheduler.step(avg_valid_metrics[0])
        
    # update our training history
    H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
    H["valid_loss"].append(avg_valid_loss.cpu().detach().numpy())
    for metric, avg_train_metric in zip(metrics, avg_train_metrics):
        H["train_" + str(metric)].append(avg_train_metric)
    for metric, avg_valid_metric in zip(metrics, avg_valid_metrics):
        H["valid_" + str(metric)].append(avg_valid_metric)
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(epoch + 1, config.NUM_EPOCHS))
    print("Train loss: {:.6f}, Valid loss: {:.4f}".format(avg_train_loss, avg_valid_loss))
    for metric, avg_train_metric, avg_valid_metric in zip(metrics, avg_train_metrics, avg_valid_metrics):
        print("Train {}: {:.6f}, Valid {}: {:.6f}".format(str(metric), avg_train_metric, str(metric), avg_valid_metric))
# display the total time needed to perform the training
end_time = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(end_time - start_time))

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["valid_loss"], label="valid_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(os.path.join(config.BASE_OUTPUT, "loss.png"))

for metric in metrics:
    # plot the training and validation metrics
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_" + str(metric)], label="train_" + str(metric))
    plt.plot(H["valid_" + str(metric)], label="valid_" + str(metric))
    plt.title("Training metric on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Scores")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(config.BASE_OUTPUT, str(metric) + ".png"))

# serialize the model to disk
MODEL_NAME = str(model) + "_" + config.DATASET_NAME + ".pth"
torch.save(model.state_dict(), os.path.join(config.BASE_OUTPUT, MODEL_NAME))

# Just for personal, to check the model
with open("mail_secret.txt", "r") as f:
    GMAIL_ID = f.readline().strip('\n')
    GMAIL_PASSWORD = f.readline().strip('\n')
smtp = smtplib.SMTP("smtp.gmail.com", 587)
smtp.starttls()
smtp.login(GMAIL_ID, GMAIL_PASSWORD)
msg = MIMEText("[MODEL] {}: training is done!".format(str(model)))
msg["Subject"] = "[imassege] Message from Deep Learning Machine"
smtp.sendmail(GMAIL_ID, GMAIL_ID, msg.as_string())
smtp.quit()