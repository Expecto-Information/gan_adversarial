import pandas as pd
from torch.utils.data import Dataset

from tqdm import tqdm
import os

from PIL import Image
from torchvision import transforms


import matplotlib.pyplot as plt
import numpy as np


class MakeDataset(Dataset):
    def __init__(self, base_dir, split, on_the_fly = True, raw_image=False,
                unlabeled_mode=False, filename_mode=False, limit=-1) :  # split = 'train_set' or 'test_set'
        super(MakeDataset, self).__init__()

        self.on_the_fly = on_the_fly

        self.limit = limit
        self.raw_image=raw_image
        self.unlabeled_mode=unlabeled_mode
        self.split = split
        

        self.data_dir = base_dir + '/' + split + '/'
        self.cats_dir = self.data_dir + 'cats/'
        self.dogs_dir = self.data_dir + 'dogs/'

        self.path_list = []
        
        self.images = []
        self.targets = []

        self.preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.process_dir(self.cats_dir, label=0)
        self.process_dir(self.dogs_dir, label=1)

        
    def process_dir(self, filedir, label):
        self.filenames = list(set(os.listdir(filedir)))
        if self.limit != -1:
            self.filenames = self.filenames[:self.limit]
        for file_name in tqdm(self.filenames, desc = f'get dataset from {filedir}...'):
            self.process_file(filedir +file_name, label)

    def process_file(self, path, label):
        if path.endswith('jpg'):
            self.path_list.append(path)
        else:
            return -1

        if self.on_the_fly == False:
            self.process_image(path)

        if self.unlabeled_mode==False:
            self.targets.append(label)

    def __len__(self):
        return len(self.path_list)
    
    def process_image(self, image_path):
        image = Image.open(image_path)
        if self.raw_image == False:
            image = self.preprocess(image)
        if self.on_the_fly == False:
            self.images.append(image)
        return image
        
    
    def show_tensor_image(self, img_tensor):
        img_array = img_tensor.numpy()
        # Transpose the array to match the shape expected by matplotlib (H x W x C)
        img_array = np.transpose(img_array, (1, 2, 0))
        plt.imshow(img_array)
        plt.show()

    def __getitem__(self, item: int):

        #pre-calculated mode or on-the-fly
        path = self.path_list[item]
        image = self.process_image(path) if self.on_the_fly == True else self.images[item]

        if self.unlabeled_mode == False:
            return image, self.targets[item]
        else:
            return image, self.path_list[item]