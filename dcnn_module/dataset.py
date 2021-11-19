# import the necessary packages
from torch.utils.data import Dataset
from dcnn_module.utils.preprocessing import resizing, random_cropping, normalize
import numpy as np
import torch
import cv2

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, resize=None, random_crop=None, normalization="z-score", channels=1, classes=1, pooling_level=4):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.channels = channels
        self.classes = classes
        self.resize = resize
        self.random_crop = random_crop
        self.normalization = normalization
        self.original_shape = None
        self.pooling_level = pooling_level
  
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
        
        # check to see if we are applying any transformations
        self.original_shape = image.shape
        if self.resize is not None:
            image = resizing(image, self.resize, interpolation=cv2.INTER_LINEAR)
            mask = resizing(mask, self.resize, interpolation=cv2.INTER_NEAREST)
        elif self.random_crop is not None:
            row_seed = np.random.randint(0, 2**32)
            col_seed = np.random.randint(0, 2**32)
            image = random_cropping(image, self.random_crop, row_seed=row_seed, col_seed=col_seed)
            mask = random_cropping(mask, self.random_crop, row_seed=row_seed, col_seed=col_seed)
        else:
            raise Exception("No augmentation is applied. Please set resize or random_crop.")
        
        image = image.astype(np.float32)
        if self.normalization is not None:
            image = normalize(image, self.normalization)
        
        # if the image has a mask, then convert the mask to the proper
        mask[mask == 255] = 1
        
        # convert the image and mask to torch tensors
        if self.channels == 1: image = np.expand_dims(image, 0)
        else: image = np.transpose(image, (2, 0, 1))
        image = torch.as_tensor(image, dtype=torch.float32)
        
        if self.classes == 1:
            mask = np.expand_dims(mask, 0)
            mask = torch.as_tensor(mask, dtype=torch.float32)
        else:
            mask = torch.as_tensor(mask, dtype=torch.int64)
        
        # return a tuple of the image and its mask
        return (image, mask)
    