import numpy as np
from torch.utils.data import Dataset
from joblib import load
from sklearn.preprocessing import LabelEncoder as label_encoder
import os
from typing import List
import torchvision.transforms as transforms
from torchvision.io import read_image
import torch
from PIL import Image
from tifffile import imread
import rasterio

class TransformDataset:
    def __init__(
        self,
        image_size:int, 
        normalize_param:List[List],
        aug:bool
    ):
        
        self.aug = aug
        self.image_size = image_size
        self.normalize_param = normalize_param
        
        
    def apply_transformations(self, image_path:str):
        # list_of_transformations =[T.RandomRotation, T.RandomAffine, T.RandomVerticalFlip, T.RandomHorizontalFlip] #Torch Vision transforms
        
        
        if image_path.endswith(".tif"):
            arr = imread(image_path)
            image = Image.fromarray(arr)
            # print(image)
            # image = rasterio.open(image_path)
            # image = image.read()
            # image = Image.fromarray(image)
        else:
            image = Image.open(image_path)
       
        transforms = self.get_composed_transform()
       
        #print(image.shape)
        transformed_image = transforms(image)
        
        return transformed_image
        
    def parse_transform(self, transform_type):
        
        method = getattr(transforms, transform_type)
        if transform_type == "RandomSizedCrop" or transform_type == "RandomResizedCrop":
            return method(self.image_size)
        elif transform_type == "CenterCrop":
            return method(self.image_size)
        elif transform_type == "Scale" or transform_type == "Resize":
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == "Normalize":
            return method(*self.normalize_param)
        else:
            return method()

    def get_composed_transform(self):
        if self.normalize_param is not None:
            if self.aug:
                transform_list = [
                    
                    "Resize",
                    "RandomResizedCrop",
                    "RandomHorizontalFlip",
                    "ToTensor",
                    "Normalize",
                ]
            else:
                transform_list = [ "Resize", "CenterCrop", "ToTensor", "Normalize"]
        else:
            if self.aug:
                transform_list = [
                    "Resize",
                    "RandomResizedCrop",
                    "RandomHorizontalFlip",
                    "ToTensor",
                ]
            else:
                transform_list = [ "Resize", "CenterCrop", "ToTensor"]

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


class MyDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split_file,
        phase,
        image_size,
        normalize_param=None
    ):

        images, labels = self.load_datafiles(split_file, phase)
        
        self.labels = labels
        self.images = images
        self.root_dir = root_dir
        
        self.phase = phase
        
        self.image_size = image_size
        
        self.normalize_param = normalize_param
        
        # print("Number of classes", np.unique(labels, return_counts=True))
       
    def load_datafiles(self, filepath, phase):

        # load the pickle file
        
        data = load(filepath)
        
        
        if phase != "all":
            
            images = data[phase + "_images"]
            labels = data[phase + "_labels"]
            return images, labels
        
        
        images = [*(data["train_images"]), *(data["test_images"]), *(data["val_images"])]
        labels = [*(data["train_labels"]), *(data["test_labels"]), *(data["val_labels"])]
        
        # print(images)
        # print(labels)
        
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image_path = os.path.join(self.root_dir, self.images[i])
        if self.phase == "test":
            aug = False
        else:
            aug = True
        # print(image_path) 
        
        transform_dataset = TransformDataset(self.image_size, self.normalize_param, aug)
        image = transform_dataset.apply_transformations(image_path)
        label = self.labels[i]
        # print(label)

        return (image, label)

class LabelWiseDataset:
    
    def __init__(self, 
        root_dir,
        split_file,
        phase,
        image_size,
        normalize_param, 
        batch_size):
        
        self.dataset = MyDataset(root_dir, split_file, phase,  image_size, normalize_param)

        self.cl_list = sorted(np.unique(self.dataset.labels).tolist())
        # print("class list",len(self.cl_list))
        
        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0,
                                  pin_memory = False)        
        for cl in self.cl_list:
            ind = np.where(np.array(self.dataset.labels) == cl)[0].tolist()
            # print("class ", cl, " records ", len(ind))
            sub_dataset = torch.utils.data.Subset(self.dataset, ind)
            # print(len(sub_dataset))
            self.sub_dataloader.append(torch.utils.data.DataLoader(
                sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        # print("inside label wise dataset, get item",i)
        # print(len(self.sub_dataloader[i]))
        # print(i)
        # item = []
        # for b, x, y in self.sub_dataloader[i]:
        item =  next(iter(self.sub_dataloader[i]))
        #     item.append((x,y))
       
        # print(len(item))
        return item

    def __len__(self):
        return len(self.sub_dataloader)
    
class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            perm = torch.randperm(self.n_classes)[:self.n_way]  
            # print(perm)
            yield perm  
            
class FewShotDataset:
    
    def __init__(self, 
        root_dir,
        split_file,
        phase,
        image_size,
        normalize_param, 
        n_way=5, 
        n_support=5, 
        n_query=16, 
        n_eposide = 100):        
        super(FewShotDataset, self).__init__()
        
        self.root_dir = root_dir
        self.split_file = split_file
        self.phase = phase
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

    def get_data_loader(self, num_workers=0):
        # print("inside few shot dataset, get data loader")
        # print("Batch Size", self.batch_size)
        dataset = LabelWiseDataset(self.root_dir, self.split_file, self.phase, self.image_size, self.normalize_param, self.batch_size)
        # print("Label wise dataset", len(dataset))
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = num_workers, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


if __name__ == "__main__":
    test_dataset = MyDataset("/Users/megha/Desktop/Plankton_Domain_Adaptation/data/kaggle",\
        "/Users/megha/Desktop/Plankton_Domain_Adaptation/data/kaggle_dataset.pkl", "train")
