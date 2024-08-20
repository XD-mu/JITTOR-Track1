from .thu_dog import ThuDog
from .food101 import Food101
from .caltech101 import Caltech101
from .animal import Animal

dataset_list = {
                "thu_dog": ThuDog,
                "food101": Food101,
                "caltech101": Caltech101,
                "animal": Animal,
                }

def build_dataset(dataset, root_path, shots):
    return dataset_list[dataset](root_path, shots)
