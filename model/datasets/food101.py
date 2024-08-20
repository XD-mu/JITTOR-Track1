import os
import random
from collections import defaultdict
from ..utils_dataset import DatasetBase

template = ['a photo of a {}, a type of food.']

class Food101(DatasetBase):

    dataset_dir = 'food101'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.anno_dir = os.path.join(self.dataset_dir, 'annotations')


        self.template = template
        self.cupl_path = './gpt4_prompts/food101.json'

        train, val, test = self.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

    # Other methods (read_data, split_trainval, save_split, read_split) would be similar to those in OxfordPets
