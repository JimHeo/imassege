from abc import ABC, abstractmethod

class DatasetType(ABC):
    @abstractmethod
    def __init__(self, dataset_name, base_path):
        pass
        
    @abstractmethod
    def parse_data(self, split=0, train_list_file=None, valid_list_file=None):
        pass