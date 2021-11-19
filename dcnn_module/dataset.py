# import the necessary packages
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, normalization='z-score', resize=None, channels=1, classes=1):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.channels = channels
        self.classes = classes
        self.normalization = normalization
        self.resize = resize
        self.original_shape = None
  
    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        image_path = self.image_paths[idx]
        # load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        if self.channels == 1:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        self.original_shape = image.shape
        # check to see if we are applying any transformations
        if self.resize is not None:
            image = cv2.resize(image, self.resize)
            mask = cv2.resize(mask, self.resize, interpolation=cv2.INTER_NEAREST)
        
        image = image.astype(np.float32)
        if self.normalization is not None:
            if self.normalization == 'z-score':
                mean = np.mean(image)
                std = np.std(image)
                if std: image = (image - mean) / std
            elif self.normalization == 'minmax':
                max_value = np.max(image)
                min_value = np.min(image)
                if max_value: image = (image - min_value) / (max_value - min_value)
            elif self.normalization == 'division':
                image = image / 255.0
        
        if self.channels == 1: image = np.expand_dims(image, 0)
        else: image = np.transpose(image, (2, 0, 1))
        mask[mask == 255] = 1
        
        image = torch.as_tensor(image, dtype=torch.float32)
        if self.classes == 1:
            mask = np.expand_dims(mask, 0)
            mask = torch.as_tensor(mask, dtype=torch.float32)
        else:
            mask = torch.as_tensor(mask, dtype=torch.int64)
        # return a tuple of the image and its mask
        return (image, mask)
    