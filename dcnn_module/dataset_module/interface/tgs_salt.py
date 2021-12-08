import os
from dcnn_module.dataset_module.interface.abstract_dataset import DatasetType
from sklearn.model_selection import train_test_split

class TGS_Salt(DatasetType):
    def __init__(self, dataset_name="TGS_Salt", base_path="dataset"):
        self.dataset_path = os.path.join(base_path, dataset_name, "train")

    def parse_data(self, split=0, train_list_file=None, valid_list_file=None):
        image_dataset_path = os.path.join(self.dataset_path, "images")
        mask_dataset_path = os.path.join(self.dataset_path, "masks")
        
        if split:
            image_list = os.listdir(image_dataset_path)
            image_list = sorted([os.path.join(image_dataset_path, image) for image in image_list])
            mask_list = os.listdir(mask_dataset_path)
            mask_list = sorted([os.path.join(mask_dataset_path, mask) for mask in mask_list])
            split_list = train_test_split(image_list, mask_list, test_size=split, random_state=42)
            (train_images, valid_images) = split_list[:2]
            (train_masks, valid_masks) = split_list[2:]
        else:
            assert train_list_file is not None and valid_list_file is not None, "If split is 0, train_list_file and valid_list_file must be specified"
            train_list_path = os.path.join(self.dataset_path, train_list_file)
            valid_list_path = os.path.join(self.dataset_path, valid_list_file)
            train_images = []
            valid_images = []
            train_masks = []
            valid_masks = []
            with open(train_list_path, 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break
                    parsed_line = line.strip().strip('\'')
                    train_images.append(os.path.join(image_dataset_path, parsed_line))
                    parsed_line = parsed_line.split('.')[0] + '.png'
                    train_masks.append(os.path.join(mask_dataset_path, parsed_line))
            with open(valid_list_path, 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break
                    parsed_line = line.strip().strip('\'')
                    valid_images.append(os.path.join(image_dataset_path, parsed_line))
                    parsed_line = parsed_line.split('.')[0] + '.png'
                    valid_masks.append(os.path.join(mask_dataset_path, parsed_line))
        train_images = sorted(train_images)
        train_masks = sorted(train_masks)
        valid_images = sorted(valid_images)
        valid_masks = sorted(valid_masks)
        
        return train_images, train_masks, valid_images, valid_masks
            
