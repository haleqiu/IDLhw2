# from torchvision.datasets import VisionDataset
import torchvision
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
import tqdm, glob, os

#### This is designed to solve the problem of loading image, while it is slow when load.
#### So we save it first to pickel and load back again
class FaceLoadImageDataset(Dataset):
    def __init__(self, datadir, data_file = "./data/FaceImage.np", target_dict = None, transform=None):
        self.transform = transform
        if os.path.exists(data_file):
            self.data = torch.load(data_file)
            self.images = self.data["image"]
            self.target_list = self.data["labels"]
            self.n_class = self.data["n_class"]
            self.target_dict = self.data["target_dict"]
        else:
            self.images = []
            file_list, target_list, class_n = self.parse_data(datadir, target_dict)
            self.file_list = file_list
            self.target_list = target_list
            self.n_class = class_n
            self.data = {"image":self.images, "labels":self.target_list, "n_class":class_n, "target_dict":self.target_dict}

            with open(data_file, 'wb') as f:
                torch.save(self.data, data_file)

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.target_list[index]
        
        if self.transform:
            self.transform(img)
        return img, label
    
    def parse_data(self, datadir, target_dict = None):
        img_list = []
        ID_list = []
        for folder in tqdm.tqdm(glob.glob(datadir+"/*")):  #root: median/
            for filename in glob.glob(folder + "/*.jpg"):
                ID_list.append(folder.split('/')[-1])
                img_list.append(filename)
                img = Image.open(filename)
                img = torchvision.transforms.ToTensor()(img)
                self.images.append(img)
        
        # construct a dictionary, where key and value correspond to ID and target
        uniqueID_list = list(set(ID_list))
        class_n = len(uniqueID_list)
        
        ## this is important
        if target_dict is None:
            self.target_dict = dict(zip(uniqueID_list, range(class_n)))
        else:
            self.target_dict = target_dict
        label_list = [self.target_dict[ID_key] for ID_key in ID_list]

        print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), len(set(label_list))))
        return img_list, label_list, class_n


#### Normal way of image dataset
class FaceImageDataset(Dataset):
    def __init__(self, datadir, target_dict = None):
        
        file_list, target_list, class_n = self.parse_data(datadir, target_dict)
        self.file_list = file_list
        self.target_list = target_list
        self.n_class = class_n

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        label = self.target_list[index]
        return img, label
    
    def parse_data(self, datadir, target_dict = None):
        img_list = []
        ID_list = []
        for folder in tqdm.tqdm(glob.glob(datadir+"/*")):  #root: median/
            for filename in glob.glob(folder + "/*.jpg"):
                ID_list.append(folder.split('/')[-1])
                img_list.append(filename)
        
        # construct a dictionary, where key and value correspond to ID and target
        uniqueID_list = list(set(ID_list))
        class_n = len(uniqueID_list)
        if target_dict is None:
            self.target_dict = dict(zip(uniqueID_list, range(class_n)))
        else:
            self.target_dict = target_dict
        label_list = [self.target_dict[ID_key] for ID_key in ID_list]

        print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), len(set(label_list))))
        return img_list, label_list, class_n

#### Evaluation dataset
class FaceImageDatasetEvl(Dataset):
    def __init__(self, datadir, target_dict = None):
        
        self.file_list = glob.glob(datadir +"/*.jpg")  
        self.target_dict = target_dict

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        idx = self.file_list[index].split("/")[-1]
        return img, idx
    

