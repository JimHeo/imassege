from dcnn_module.dataset_module.interface.blur_detection import BlurDetection
from dcnn_module.dataset_module.interface.tgs_salt import TGS_Salt

class DatasetParser:
    def __init__(self, dataset_name, base_path="dataset"):
        self.dataset_name = dataset_name
        self.base_path = base_path
    
    def connection(self, split=0, train_list_file=None, valid_list_file=None):
        if self.dataset_name == "BlurDataset":
            connector = BlurDetection(self.dataset_name, self.base_path)
        elif self.dataset_name == "TGS_Salt":
            connector = TGS_Salt(self.dataset_name, self.base_path)
        else:
            raise NotImplementedError
        
        return connector.parse_data(split, train_list_file, valid_list_file)