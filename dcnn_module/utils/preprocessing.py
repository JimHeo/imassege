from math import ceil, floor
import numpy as np
import cv2

def resizing(image, size, interpolation=cv2.INTER_LINEAR):
    return cv2.resize(image, size[::-1], interpolation=interpolation)

def random_cropping(image, size, row_seed, col_seed):
    np.random.seed(row_seed)
    row_start = np.random.randint(0, image.shape[0] - size[0] + 1)
    np.random.seed(col_seed)
    col_start = np.random.randint(0, image.shape[1] - size[1] + 1)
    image = image[row_start:row_start + size[0], col_start:col_start + size[1]]
    
    return image

def cropping_to_fit(image, level=4):
    row, col = image.shape[0], image.shape[1]
    row_start = floor((row % 2**level) / 2)
    row_end = row - ceil((row % 2**level) / 2)
    col_start = floor((col % 2**level) / 2)
    col_end = col - ceil((col % 2**level) / 2)
    image = image[row_start:row_end, col_start:col_end]
    
    return image

def padding_to_fit(image, origin_shape):
    origin_row, origin_col = origin_shape[0], origin_shape[1]
    image_row, image_col = image.shape[0], image.shape[1]
    image = cv2.copyMakeBorder(image,
                              top=floor((origin_row - image_row) / 2), 
                              bottom=ceil((origin_row - image_row) / 2),
                              left=floor((origin_col - image_col) / 2),
                              right=ceil((origin_col - image_col) / 2),
                              borderType=cv2.BORDER_REFLECT_101)
    
    return image

def normalize(image, type="division"):
    if type == "z-score":
        mean = np.mean(image)
        std = np.std(image)
        if std: image = (image - mean) / std
    elif type == "minmax":
        max_value = np.max(image)
        min_value = np.min(image)
        if max_value: image = (image - min_value) / (max_value - min_value)
    elif type == "division":
        image = image / 255.0
    else:
        raise Exception("Normalization-Type is one of [\"z-score\", \"minmax\", \"division\"]")
        
    return image